"""
This script takes as input a treebank file in conllu format, a configuration (ud, sud or basic),
a list of features and their values to be added to the treebank, the path to the output file 
and the level at which the features and their values should be added (token, sentence or both).

It adds the features and their values to the treebank and saves the updated treebank to a file.

Example of usage for adding features and values to all sentences/tokens in the treebank:
python3 add_feat_value.py -path Treebanks/french_pud.conllu --config ud --features_values language=French genre=news -o Treebanks/french_pud_updated.conllu --level both

Example of usage for adding features and values to nodes that match a grewmatch query:

python3 add_feat_value.py -path Treebanks/french_pud.conllu --config ud --features_values Case=Dative -o Treebanks/french_pud_updated.conllu --query

"""
import grewpy
from grewpy import Corpus, CorpusDraft, Request
import argparse
import pathlib


def add_meta(draft, liste_feats_values):
    """
    This function takes as arguments a corpus(Corpus object), a language(string) and a genre(string)
    and adds the language and genre to the metadata of each sentence in the corpus.
    """
    for feat, value in liste_feats_values:
        for sent_id in draft:
            draft[sent_id].meta[feat.lower()] = value

    return draft


def add_misc(draft, liste_feats_values):
    """
    This function takes as arguments a corpus(Corpus object), a language(string) and a genre(string)
    and adds the language and genre to the misc field of each token in the corpus.
    """
    for feat, value in liste_feats_values:
        for sent_id in draft:
            for token in draft[sent_id].features:
                if token != "0":
                    draft[sent_id].features[token][feat.capitalize()] = value

    return draft

def add_query_feat_value(treebank_path, features_values, config, output):
    """
    This function takes as arguments the path to the treebank file, a list of features and their values to be added to the treebank,
    a configuration (ud or sud, ud by default) and the path to the output file. It also prompts the user to enter a grewmatch query
    """
    print("WARNING! Only the nodes that match the query will be annotated. If you want to annotate all the nodes, please rerun the script without the query option.")
    query = input("Enter the grewmatch query(e.g. X-[nsubj]->Y): ")
    node_to_annotate = input("Enter the name used for the node to annotate in your query (eg. X): ")

    if node_to_annotate not in query:
        raise ValueError(f"The node to annotate {node_to_annotate} is not present in the query {query}. Please make sure to use the same name for the node to annotate in the query.")
    
    liste_feats_value = []
    for feat_value in features_values:
        if "=" not in feat_value:
            raise ValueError(f"Invalid format for features_values: {feat_value}. Format should be feature=value.")
        liste_feats_value.append(tuple(feat_value.split("=")))

    grewpy.set_config(config)  
    corpus = Corpus(treebank_path)  
    draft = CorpusDraft(corpus)

    req = Request(query)
    matched_patterns = corpus.search(req)
    matched_sentences = {}
    for i in range(len(matched_patterns)):
        matched_sentences[matched_patterns[i]['sent_id']] = matched_patterns[i]["matching"]["nodes"][node_to_annotate]

    for feat, value in liste_feats_value:
        for sent_id, token in matched_sentences.items():
            draft[sent_id].features[token][feat] = value

    conll_string = draft.to_conll()
    with open(output, "w") as file:
        file.write("# global.columns = ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC \n")
        file.write(conll_string)
    print(f"The node {node_to_annotate} in the treebank has been annotated with the features and their values.")

def add_weight(treebank_path, config, output):
    grewpy.set_config(config)
    corpus = Corpus(treebank_path)
    draft = CorpusDraft(corpus)

    for i in range(len(draft)):
        sentence = draft[i]
        left, right = {i: i for i in sentence}, {i: i for i in sentence}
        todo = [i for i in sentence]
        for key, value in sentence.sucs.items():
            if key != "0":
                sentence.features[key]["Weight"] = str(len(value)) 
        sentence.sucs = {i : sentence.sucs.get(i,[]) for i in sentence}
        while todo:
            n = todo.pop(0)
            for s, _ in sentence.sucs[n]:
                if sentence.lower(left[s], left[n]):
                    left[n] = left[s]
                    todo.append(n)
                if sentence.greater(right[s], right[n]):
                    right[n] = right[s]
                    todo.append(n)
        for i in sentence:
            sentence.features[i]["Left_span"] = left[i]
            sentence.features[i]["Righ_span"]= right[i]
            if sentence.features[i].get("Weight") == None:
                sentence.features[i]["Weight"] = "0"

    conll_string = draft.to_conll()
    with open(output, "w") as file:
        file.write("# global.columns = ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC \n")
        file.write(conll_string)
    print(f"The treebank has been updated with the number of dependents each token has, as well as the right and left span.")
    return output 


def parse_args():
    """
    This function parses the arguments given to the script.
    """
    parser = argparse.ArgumentParser(
        description="Add features and their values to each sentence or/and tokens in a treebank.",
    )

    parser.add_argument(
        "-path",
        "--treebank_path",
        required=True,
        help="Path to the treebank file.",
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
        "--features_values",
        "-fv",
        nargs="+",
        action="extend",
        type=str,
        help="List of features and their values to be added to the treebank. Format : feature1=value1 feature2=value2 ...",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=pathlib.Path,
        required=True,
        help="Path to the output file, must be conllu format.",
    )

    parser.add_argument(
        "--level",
        "-l",
        choices=["token", "sentence", "both"],
        default="both",
        nargs="?",
        help="Level at which the features and their values should be added. Optional argument, default is both.",
    )

    parser.add_argument(
        "--query",
        "-q",
        action="store_true",
        help="Grewmatch query to select the nodes to which the features and their values should be added.",
    )

    parser.add_argument(
        "--weight",
        "-w",
        action="store_true",
        help="Annotate each token with the number of dependents it has."
    )

    return parser.parse_args()

def main(treebank_path, config, features_values, output, level):
    # Check if the format of the features and their values is correct (feature=value)
    liste_feats_value = []
    for feat_value in features_values:
        if "=" not in feat_value:
            raise ValueError(f"Invalid format for features_values: {feat_value}. Format should be feature=value.")
        liste_feats_value.append(tuple(feat_value.split("=")))

    grewpy.set_config(config)  
    corpus = Corpus(treebank_path)  
    draft = CorpusDraft(corpus)

    # Map the level given as argument to the corresponding function
    level_function_map = {
        "sentence": add_meta,
        "token": add_misc,
        "both": lambda draft, feats: add_misc(add_meta(draft, feats), feats),
    }

    # Add features and their values to the treebank, based on the chosen level
    draft = level_function_map[level](draft, liste_feats_value)

    # Save the updated treebank to a file
    conll_string = draft.to_conll() 
    with open(output, "w") as file:
        file.write(conll_string)
    print(f"The treebank has been updated with the features and their values.")
    return output

def main_cli():
    args = parse_args()
    if args.query:
        add_query_feat_value(args.treebank_path, args.features_values, args.config, args.output)
    elif args.weight:
        add_weight(args.treebank_path, args.config, args.output)
    else:
        main(args.treebank_path, args.config, args.features_values, args.output, args.level)

if __name__ == "__main__":
    main_cli()
