import argparse
import yaml
import json
import numpy as np
import scipy.stats

from sklearn import tree
from grex2.grex.data import extract_data
from grex2.grex.utils import FeaturePredicate, pattern_to_request

import pyximport
pyximport.install()
import grex2.grex.features


def parents_from_dtree(T):
    """Build a dictionary child : (parent, sign) where sign is the split decision (0,1)."""
    parents = dict()
    for node in range(T.node_count):
        left = T.children_left[node]
        right = T.children_right[node]
        if left >= 0: parents[left] = (node, 0)
        if right >= 0: parents[right] = (node, 1)
    return parents


def branch_from_parents(node, parents):
    """Compute the branch from node n to the root of the decision tree."""
    branch = []
    while node != 0:
        node, split = parents[node] # split = 0, 1 (resp. left, right)
        branch.append((node, split))
    branch.reverse()
    return branch


def pattern_from_dtree(T, branch, feature_names):
    """Build a rule pattern from a branch of the decision tree."""
    attributs = [(n, feature_names[T.feature[n]], decision) for n, decision in branch]
    pattern = [f"{decision}:{att}" for _, att, decision in attributs] 
    return pattern


if __name__ == "__main__":
    cmd = argparse.ArgumentParser()
    cmd.add_argument('data', metavar='F', type=str, nargs='+', help='data')
    cmd.add_argument("--output", type=str, required=True)
    cmd.add_argument("--patterns", type=str, required=True)
    cmd.add_argument("--config", type=str, default="ud")
    cmd.add_argument("--grew", action="store_true", help="return a grew request")
    cmd.add_argument("--min-samples_leaf", type=int, default=5)
    cmd.add_argument("--node_impurity", type=float, default=0.15)
    cmd.add_argument("--threshold", type=float, default=1e-1)
    cmd.add_argument("--max-depth", type=int, default=12)
    cmd.add_argument("--only-leaves", action="store_true", help="Path only to leaves")

    args = cmd.parse_args()

    with open(args.patterns) as instream:
        config = yaml.load(instream, Loader=yaml.Loader)

    scope = config["scope"]
    conclusion = config.get("conclusion", None)
    conclusion_meta = config.get("conclusion_meta", None)

    templates = FeaturePredicate.from_config(config["templates"])
    feature_predicate = FeaturePredicate.from_config(config["features"], templates=templates)

    print("Loading dataset...", flush=True)
    include_metadata = any('meta' in k for k in config.get('features', {}).get('sentence', {}))
    data = extract_data(args.data, scope, conclusion, conclusion_meta, feature_predicate, args.config, include_metadata)

    # quick checks
    if len(data) == 0:
        raise RuntimeError("Patterns resulted in empty dataset")
    num_positive = sum(sentence["output"] for sentence in data)
    if num_positive == 0:
        raise RuntimeError("The conclusion does not appear in the dataset")
    if num_positive == len(data):
        raise RuntimeError("The conclusion always appears in the dataset")

    data_inputs = list()
    data_outputs = list()
    for sentence in data:
        data_inputs.append(sentence["input"])
        data_outputs.append(sentence["output"])

    print("Number of occurences of the conclusion: %i / %i" % (num_positive, len(data)))

    print("Extracting features", flush=True)
    feature_set = grex2.grex.features.FeatureSet()
    feature_set.add_feature(grex2.grex.features.AllSingletonFeatures())

    try:
        feature_set.init_from_data(data_inputs)
        feature_names = [f for f in feature_set.features[0].get_all_names()]
        X = feature_set.build_features(data_inputs, sparse=True)
        if X.shape[1] == 0:
            raise RuntimeError("Empty feature list!")
    except RuntimeError:
        RuntimeError("There was an error during feature extraction")

    # build targets
    y = np.empty((len(data),), dtype=np.int_)
    for i, v in enumerate(data_outputs):
        assert v in [0, 1]
        y[i] = v

    extracted_rules = dict()
    extracted_rules['scope'] = scope
    if conclusion_meta:
        meta = ",".join(f"{k}={v}" for k, v in conclusion_meta.items())
        extracted_rules['conclusion'] = f"{conclusion},{meta}" if conclusion else meta
    else:
        extracted_rules['conclusion'] = conclusion or ""
    extracted_rules["data_len"] = len(data)
    extracted_rules["n_yes"] = num_positive

    # extract rules
    rules = []
    clf = tree.DecisionTreeClassifier(criterion="entropy", 
                                    min_samples_leaf=args.min_samples_leaf, 
                                    max_depth=args.max_depth,
                                    random_state=42)
    clf.fit(X, y)
    T = clf.tree_
    dtree_parents = parents_from_dtree(T)

    selected_nodes = []
    if args.only_leaves:
        for n in range(T.node_count):
            if T.children_left[n] == T.children_right[n] == -1: #is leaf
                decision = int(np.argmax(T.value[n]))
                selected_nodes.append((decision, n))
    else:
        for n in range(T.node_count):
            if T.impurity[n] < args.threshold:
                if np.argmax(T.value[n]):
                    selected_nodes.append((1, n))
                else:
                    selected_nodes.append((0, n))

    for decision, n in selected_nodes:
        branch = branch_from_parents(n, dtree_parents)
        pattern = pattern_from_dtree(T, branch, feature_names)
        rule = pattern_to_request(pattern, scope) if args.grew else pattern

        n_matched = int(T.n_node_samples[n])
        n_pattern_positive_occurence = n_matched*T.value[n][0,1]
        n_pattern_negative_occurence = n_matched*T.value[n][0,0]

        node_path = clf.decision_path(X)
        matched_samples = node_path[:, n].toarray().flatten()

        expected = (n_matched * num_positive) / len(data)
        delta_observed_expected = n_pattern_positive_occurence - expected

        coverage_p = n_pattern_positive_occurence / n_matched
        coverage_not_p = n_pattern_negative_occurence / n_matched

        observed = np.array([n_pattern_positive_occurence, n_pattern_negative_occurence])
        #expected = np.array([expected, n_matched - expected])
        expected = np.array([n_matched / 2, n_matched / 2])

        if np.any(expected == 0):
            chi2_val = 0.0
            p_value = 1.0
            cramers_phi = 0.0
        else:
            chi2_val, p_value = scipy.stats.chisquare(f_obs=observed, f_exp=expected)
            cramers_phi = np.sqrt(chi2_val / n_matched)

        if decision:
            coverage = (n_pattern_positive_occurence / num_positive) * 100
            precision = (n_pattern_positive_occurence / n_matched) * 100
            ratio = (coverage_p / coverage_not_p) if (coverage_p != 0 and coverage_not_p != 0) else 0
        else:
            coverage = (n_pattern_negative_occurence / (len(data) - num_positive)) * 100
            precision = (n_pattern_negative_occurence / n_matched) * 100
            ratio = (coverage_not_p / coverage_p) if (coverage_p != 0 and coverage_not_p != 0) else 0
            
        rules.append({
            "pattern": str(rule),
            "n_pattern_occurences": n_matched,
            "n_pattern_positive_occurences": int(n_pattern_positive_occurence),
            "n_pattern_negative_occurrences": int(n_pattern_negative_occurence),
            "decision": "yes" if decision else "no",
            "coverage": coverage,
            "coverage_q_in_p": coverage_p,
            "coverage_not_q_in_p": coverage_not_p,
            "precision": precision,
            "ratio_coverage_in_p": ratio,
            "delta": delta_observed_expected,
            "g-statistic": chi2_val,
            "p-value": p_value,
            "cramers_phi": cramers_phi
        })

    extracted_rules['rules'] = rules[::-1]

print("Done.", flush=True)
with open(args.output, 'w') as out_stream:
    json.dump(extracted_rules, out_stream)
