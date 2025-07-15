import argparse
import pathlib
import grewpy

def remove_punctuation(corpus):
    s = """
    strat main { Onf(rm) }
    rule rm {
        pattern { X[upos=PUNCT] }
        without { Y[upos]; X < Y}
        commands {del_node X}
    }
    """
    grs = grewpy.GRS(s)
    corpus = grs.apply(corpus)
    return corpus

def remove_MISC(conll):
    res = []
    lines = conll.split("\n")
    for line in lines:
        columns = line.split("\t")
        if len(columns) > 9:
            columns[9] = "_"
        line_str = "\t".join(columns)
        res.append(line_str)
    return "\n".join(res)

def compute_length(corpus):

    def get_length_range(value):
        # bins = [
        #     (1, 1, "1"),
        #     (1, 3, "1-3"),
        #     (1, 7, "1-7"),
        # ]
        bins = [
            (1, 1, "1"),
            (2, 3, "2-3"),
            (4, 7, "4-7"),
            (8, float('inf'), "8+")
        ]

        for lower, upper, label in bins:
            if lower <= value <= upper:
                return label
        return None

    # def get_length(value):
    #     res = []
    #     bins = [
    #         (2, float('inf'), "2+"),
    #         (4, float('inf'), "4+"),
    #         (8, float('inf'), "8+")
    #     ]
    #     for lower, upper, label in bins:
    #         if lower <= value <= upper:
    #             res.append(label)
    #     return res
    
    def length_(token_id, sucs):
        if token_id in sucs:
            return 1 + sum(length_(child[0], sucs) for child in sucs[token_id])
        else:
            return 1

    draft = grewpy.CorpusDraft(corpus)
    for sent_id, graph in draft.items():
        successors = graph.sucs
        for token_id in graph:
            if token_id == "0":
                continue
            
            length = length_(token_id, successors)
            length_bin = get_length_range(length)
            if length_bin:
                draft[sent_id][token_id].update({'len': length_bin})

    res = grewpy.Corpus(draft)
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='The corpus to process', required=True)
    args = parser.parse_args()

    paths = pathlib.Path(args.input).glob("*.conllu")
    for p in paths:
        base_directory = p.parent
        filename = p.name
        
        corpus = grewpy.Corpus(p.as_posix())
        corpus_without_punct = remove_punctuation(corpus)
        conll = corpus_without_punct.to_conll()
        corpus_without_misc = grewpy.Corpus(remove_MISC(conll))
        corpus_with_length = compute_length(corpus_without_misc)
        directory = base_directory / "preprocessed_data"
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory.as_posix() + "/" + filename, "w") as f:
            f.write(corpus_with_length.to_conll())
