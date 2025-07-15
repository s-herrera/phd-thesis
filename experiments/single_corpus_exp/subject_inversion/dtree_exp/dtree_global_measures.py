import grewpy
import numpy as np
import json
from evaluation.global_eval_measures import global_coverage, average_maximum_precision, redundancy
from univariate.univariate import compute_rule_precision

def build_X_y_selected(path, scope_pattern, patterns):
    corpus = grewpy.Corpus(path)
    scope_req = grewpy.Request(scope_pattern)
    scope = corpus.search(scope_req)
    scope_matches = [(match['sent_id'], match['matching']['nodes']['X'], match['matching']['nodes']['Y']) for match in scope]
    X_selected = np.zeros((len(scope_matches), len(patterns)))
    y = np.array([1 if int(m[1]) < int(m[2]) else 0 for m in scope_matches])
    for col_idx, pattern in enumerate(patterns):
        pattern_matches = corpus.search(grewpy.Request(pattern))
        pattern_matches_set = set(
            (match['sent_id'], match['matching']['nodes']['X'], match['matching']['nodes']['Y'])
            for match in pattern_matches
        )
        for row_idx, match in enumerate(scope_matches):
            X_selected[row_idx, col_idx] = 1 if match in pattern_matches_set else 0
    return X_selected, y



paths = [
    "../../../../results/results_subject_inversion_dtree_depth8.json",
    "../../../../results/results_subject_inversion_dtree_depth14.json"
]

corpus_path = "../../../../data/single_corpus_data/SUD_French-GSD-r2.15/preprocessed_data"
scope_req = grewpy.Request("pattern {X-[1=subj]->Y}")

res = {}
for path in paths:
    with open(path) as f:
        data = json.load(f)
    model = path.split("_")[4]
    scope_req = grewpy.Request(data['scope'])
    rules = data['rules']
    patterns = [r['pattern'] for r in rules]
    X_selected, y = build_X_y_selected(corpus_path, scope_req, patterns)
    npz_data = np.load("results_subject_inversion_dtree_depth8_data.npz", allow_pickle=True)

    selected_precisions = [r['precision'] for r in rules]
    selected_coverages = [r['coverage'] for r in rules]
    selected_signs = [1 if r['decision'] == 'yes' else 0 for r in rules]

    selected_rules = compute_rule_precision(X_selected, y, patterns)

    global_coverage_score, uncovered_occs = global_coverage(X_selected, y, selected_signs)
    redudancy_score = redundancy(X_selected)
    amp, pos_amp, neg_amp, rule_zero_amp, (pos_base_precision, neg_base_precision) = average_maximum_precision(selected_rules, y)
    avg_lenght = np.mean([len(name.split("\n"))-1 for name in patterns])
    
    res[model] = {
        "global_coverage" : global_coverage_score,
        "redudancy" : redudancy_score,
        "amp" : amp,
        "pos_amp" : pos_amp,
        "neg_amp" : neg_amp,
        "rule_zero_amp" : rule_zero_amp,
        "pos_base_precision" : pos_base_precision,
        "neg_base_precision" : neg_base_precision,
        "gain" : amp - rule_zero_amp,
        "uncovered_occs" : uncovered_occs[0],
        "n_features": len(patterns),
        "avg_precision": np.mean(selected_precisions),
        "avg_coverage": np.mean(selected_coverages),
        "avg_lenght": avg_lenght
    }

with open("dtree_global_measures.json", "w") as f:
    json.dump(res, f)