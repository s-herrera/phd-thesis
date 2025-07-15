"""
This script combines two or more treebanks, keeping n matches from each 
and annotating them automatically with the language of the treebank. 
They are then saved in a new file.

Example:
python3 combine_treebanks.py -p /path/to/english/treebank token language=English genre=news -p /path/to/french/corpus sentence language=French genre=fiction -p /path/to/spanish/corpus both language=Spanish genre=medical -n 1000 --random -o /path/to/output/file.conllu -pos
"""

from add_feat_value import main as add_feat_value
from conllu import parse_incr
from grewpy import Corpus, CorpusDraft
import grewpy
import tempfile
import argparse
import os
import random
from grewpy import GRSDraft, GRS
from collections import defaultdict


def get_level_and_features(group):
    """Extract level and features from the group."""
    if not isinstance(group, list):
        raise ValueError("group must be a list of strings")
    path = group[0]
    level = group[1] if group[1] in ["sentence", "token", "both"] else "both"
    features = group[2:] if group[1] in ["sentence", "token", "both"] else [group[1]]
    return path, level, features

def get_random_n_matchs(matches):
    my_matches = {}
    for i in range(len(matches)):
        if matches[i]["sent_id"] not in my_matches:
            my_matches[matches[i]["sent_id"]] = [matches[i]['matching']]
        else:
            my_matches[matches[i]["sent_id"]].append(matches[i]['matching'])

    keys = list(my_matches.keys())
    random.shuffle(keys)
    my_matches_randomised = {key: my_matches[key] for key in keys}
    new_matches = []
    for key, value in my_matches_randomised.items():
        for m in value:
            new_matches.append({'sent_id': key, 'matching': m})
    return new_matches

def get_sentence_length(ith_draft, pos):
    if pos:
        nb_verbs, nb_nouns, nb_adjs = 0, 0, 0
        sentence = ith_draft.features
        for _, feats in sentence.items():
            if "upos" in feats:
                if feats["upos"] == "VERB":
                    nb_verbs += 1
                elif feats["upos"] == "NOUN":
                    nb_nouns += 1
                elif feats["upos"] == "ADJ":
                    nb_adjs += 1
        sentence_length = nb_verbs + nb_nouns + nb_adjs
    else:
        sentence_length = len(ith_draft) - 1
    return sentence_length

def get_iqr(treebanks_details):
    all_sentences = []
    for treebank_path, sentences in treebanks_details.items():
        all_sentences.extend(sentences.values())
    all_sentences.sort()
    q1 = all_sentences[len(all_sentences) // 4]
    q3 = all_sentences[3 * len(all_sentences) // 4]
    iqr = q3 - q1
    lower_bound = abs(q1 - 1.5 * iqr)
    upper_bound = abs(q3 + 1.5 * iqr)
    return lower_bound, upper_bound

def validate_path(path):
    """Check if the path exists and is readable."""
    if not os.path.exists(path):
        raise ValueError(f"The path {path} does not exist.")
    if not os.access(path, os.R_OK):
        raise ValueError(f"The file {path} is not readable.")
    

def get_clean_sentences(treebanks_to_concat, corpus_drafts, pos, iqr_selection):
    treebanks_details = {}
    for treebank_path, _ in treebanks_to_concat.items():
        treebanks_details[treebank_path] = {}
        corpus = corpus_drafts[treebank_path][0]
        draft = corpus_drafts[treebank_path][1]
        sent_ids = corpus.get_sent_ids()
        for i in range(len(draft)):
            treebanks_details[treebank_path][sent_ids[i]] =  get_sentence_length(draft[i], pos)
    
    lower_bound, upper_bound = get_iqr(treebanks_details)

    if pos:
        print("The length of the sentences is measured as the number of verbs, nouns and adjectives.")
        print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")
        print("Sentences with length outside this range will not be included in the output.")
    else:
        print("The length of the sentences is measured as the number of tokens.")
        print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")
        print("Sentences with length outside this range will not be included in the output.")
    print("----------------------------------------------------------\n ")

    clean_tb_sentids = {}
    for treebank_path, sentences in treebanks_details.items():
        sentids_to_add = []
        for sent_id, sentence_length in sentences.items():
            if iqr_selection:
                if sentence_length > lower_bound and sentence_length < upper_bound:
                    sentids_to_add.append(sent_id)
            else:
                sentids_to_add.append(sent_id)
        clean_tb_sentids[treebank_path] = sentids_to_add
    
    return clean_tb_sentids

def remove_enhanced_deps(corpus):
    s = """
    package UD2bUD {
    rule enh { % remove enhanced relations
        pattern { e:N -[enhanced=yes]-> M }
        commands { del_edge e}
    }

    rule empty { % remove empty nodes
        pattern { N [wordform=__EMPTY__, textform=_] }
        commands { del_node N }
    }
    }

    strat main { Onf(UD2bUD) }
    """
    grs_draft = GRSDraft(s)
    grs = GRS(grs_draft)
    return corpus.apply(grs)

def remove_punctuation(corpus):
    s = """
    package UD2bUD {
    rule enh { % remove punct relations
        pattern { e:N -[punct]-> M }
        commands { del_edge e}
    }

    rule empty { % remove punct nodes
        pattern { N [upos="PUNCT"] }
        commands { del_node N }
    }
    }

    strat main { Onf(UD2bUD) }
    """
    grs_draft = GRSDraft(s)
    grs = GRS(grs_draft)
    return corpus.apply(grs)

def add_corpus_to_file(
    treebanks_to_concat, corpus_drafts, output_path, n, request, config, pos, iqr_selection, random_sampling=False
    ):
    """
    This function takes as arguments a list of paths to treebanks, a path to the output file, a list of feature values and a boolean.
    It reads the treebanks, adds the feature values to the metadata and the misc field of each token in the treebanks,
    combines the treebanks and writes them to the output file.
    """
    clean_tb_sentids = get_clean_sentences(treebanks_to_concat, corpus_drafts, pos, iqr_selection)

    with open(output_path, "w") as out_file:
        for treebank_path, annotation_details in treebanks_to_concat.items():
            print(f"Processing treebank at {treebank_path}...")
            parts = treebank_path.split("/")
            filename = parts[-1]
            filename_parts = filename.split("_")
            language = filename_parts[1] if len(filename_parts) > 1 else filename_parts[0]
            print(f"Language: {filename}") 
            
            pud = "pud" in filename
            if pud: # PUD treebanks have the same IDs for each language
                print(f"A PUD treebank...")
                temp_file = tempfile.NamedTemporaryFile(delete=False)  
                temp_file_path = temp_file.name  
                with open(treebank_path, "r") as data_file, open(temp_file_path, "w") as temporary_file:
                    data_file.seek(0)  # Reset the file pointer to the beginning of the file
                    for tokenlist in parse_incr(data_file):
                        sent_id = tokenlist.metadata["sent_id"]
                        tokenlist.metadata["sent_id"] = f"{sent_id}-{language}"
                        temporary_file.write(tokenlist.serialize() + '\n')  # Write the modified data to the output file
                treebank_path = temporary_file.name

            corpus = corpus_drafts[treebank_path][0]
            draft = corpus_drafts[treebank_path][1]

            matches = corpus.search(grewpy.Request(request))
            corpus.clean()
            print("all matches:", len(matches))

            # matches to only include those with sent_id in clean_tb_sentids[treebank_path]
            clean_matches = [m for m in matches if m["sent_id"] in clean_tb_sentids[treebank_path]]
            print("matches in the IQR bounds:", len(clean_matches))

            if len(clean_matches) < n:
                needed = n - len(clean_matches)
                remaining_matches = [m for m in matches if m not in clean_matches]
                if needed > len(remaining_matches):
                    raise ValueError(f"Not enough unique matches to reach {n} for {treebank_path}.")
                extra_matches = random.sample(remaining_matches, needed)
                clean_matches.extend(extra_matches)
            else:
                matches_by_sent_id = defaultdict(list)
                for m in clean_matches:
                    matches_by_sent_id[m['sent_id']].append(m)

                sent_id_counts = sorted(matches_by_sent_id.items(), key=lambda x: len(x[1]), reverse=True)
                selected_sent_ids = []
                total_matches = 0

                for sent_id, matches in sent_id_counts:
                    if total_matches + len(matches) <= n:
                        selected_sent_ids.append(sent_id)
                        total_matches += len(matches)
                    elif total_matches < n:
                        selected_sent_ids.append(sent_id)
                        total_matches += len(matches)
                        break
                    else:
                        break

                clean_matches = []
                for sid in selected_sent_ids:
                    clean_matches.extend(matches_by_sent_id[sid])

            n_matches = len(clean_matches)
            print(f"Final selection: {n_matches} matches from {n} sentences")
            if random_sampling:
                matches = get_random_n_matchs(matches)

            print("Selecting sentences...")
            sent_ids = [m["sent_id"] for m in clean_matches]

            tmp = tempfile.NamedTemporaryFile()
            with open(tmp.name, "a") as f:
                if list(treebanks_to_concat.keys())[0] == treebank_path:
                    f.write(
                        "# global.columns = ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC \n"
                    )
                for sent_id in sent_ids:
                    conll_string = draft[sent_id].to_conll()
                    f.write(conll_string)
                    f.write("\n")
            print(
                f"Adding features {annotation_details[1]} at {annotation_details[0]} level to treebank at {treebank_path}..."
            )
            annotated_file = add_feat_value(
                tmp.name, config, annotation_details[1], tmp.name, annotation_details[0]
            )
            with open(annotated_file, "r") as annotated_file:
                out_file.write(annotated_file.read())
            print(f"Finished processing treebank at {treebank_path}.")
            print("---")
    return output_path

def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine two or more treebanks and add features and their values to them.",
    )

    parser.add_argument(
        "-p",
        "--path_level_and_features",
        required=True,
        nargs="+",
        action="append",
        help="Path to the treebank, its level (optional, default is 'both'), followed by feature=value pairs. Each group should be provided as consecutive arguments after -p. For example: -p /path/to/english/treebank level=sentence language=English genre=news -p /path/to/french/corpus level=token language=French genre=fiction",
    )

    parser.add_argument(
        "-n",
        "--n_matches",
        required=True,
        default=-1,
        type=int,
        help="Number of matchs to be kept from each treebank.",
    )

    parser.add_argument(
        "-req",
        "--request",
        required=True,
        type=str,
        help="Grew request",
    )

    parser.add_argument(
        "-r" ,"--random",
        action="store_true",
        help="Randomly select n sentences from each treebank.",
    )

    parser.add_argument(
        "-o",
        "--output_path",
        required=True,
        help="Path to the output file.",
    )

    parser.add_argument(
        "--config",
        "-c",
        choices=["ud", "sud", "basic"],
        default="ud",
        type=str,
        nargs="?",
        help="Configuration of the treebank. Default is ud.",
    )

    parser.add_argument(
        "-pos",
        "--pos",
        action="store_true",
        help="Clean sentences based on their length measured as the number of verbs, nouns and adjectives if stored true, otherwise the cleaning is based on the number of tokens.",
    )

    parser.add_argument(
        "-iqr",
        "--iqr-selection",
        action="store_true",
        help="Whether to the IQR selection",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    grewpy.set_config(args.config)
    treebanks_to_concat = {} # contains path : (level, feature)
    path_level_and_features = []
    corpus_drafts = {}
    for group in args.path_level_and_features:
        path, level, features = get_level_and_features(group)
        path_level_and_features.append([path, level, features])
        validate_path(path)
        treebanks_to_concat[path] = (level, features)
        corpus = remove_enhanced_deps(remove_punctuation(Corpus(path)))
        draft = CorpusDraft(corpus)
        corpus_drafts[path] = (corpus, draft)
        print("Made corpus and draft for " + path)

    random.seed(42)
    add_corpus_to_file(
        treebanks_to_concat, corpus_drafts, args.output_path, args.n_matches, args.request, args.config, args.pos, args.iqr_selection, args.random
    )
if __name__ == "__main__":
    main()
