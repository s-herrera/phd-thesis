import numpy as np
import argparse
import json
from evaluation import global_coverage, redundancy


def greedy_forward_global_score_selection(X, y, rules, rule_zero=True, rule_filter='all'):
    """
    Greedy forward selection of rules based on global score improvement.
    (Current implementation optimizes mean score over ALL examples)

    At each iteration, the algorithm evaluates all (remaining) rules based on the rule_type_filter.
    If a rule improves the global score when added to the current set, it is selected.
    The process stops if no rule can improve the score.

    Args:
        X: Feature matrix.
        y: Target vector.
        rules (list): List of rules, where each rule is (name, vec, score, sign).
                      'sign' is True for positive-oriented rules, False for negative-oriented.
        rule_zero (bool): Whether to apply a baseline rule_zero initialization.
        rule_type_filter (str): Specifies which rules to consider.
                                'all': Considers all rules.
                                'positive_only': Considers only rules where sign is True.
                                'negative_only': Considers only rules where sign is False.
    """

    #checks
    assert rule_filter in ['all', 'positive_only', 'negative_only']
    print("Greedy Selection", flush=True)

    n_occs = X.shape[0]
    pos_mask = (y == 1)
    neg_mask = (y == 0)
    current_assign = np.zeros(n_occs)
    global_score = 0.0

    # baseline probabilities/scores for rule_zero and uncovered counts
    base_pos_prob = np.mean(y) 
    base_neg_prob = np.mean(np.logical_not(y))

    if rule_zero:
        # baseline probabilities only to respective classes
        current_assign[pos_mask] = base_pos_prob
        current_assign[neg_mask] = base_neg_prob
        # global_score is the mean of these initial assignments across all occurrences
        global_score = np.mean(current_assign)

    selected_rules = [] 

    improved = True
    while improved:
        improved = False
        best_gain = 0.0
        best_rule_name = None 
        best_assign = None    

        for name, vec, score, sign in rules:

            # filter rules
            if rule_filter == 'positive_only' and not sign:
                continue
            if rule_filter == 'negative_only' and sign:
                continue

            # check if rule already selected by name
            is_already_selected = any(selected_name == name for selected_name in selected_rules)
            if is_already_selected:
                continue
                
            trial_assign = current_assign.copy()
            # Ensure vec is dense (handle both numpy arrays and scipy sparse matrices)
            if hasattr(vec, "todense"):
                vec_dense = np.asarray(vec.todense()).squeeze()
            else:
                vec_dense = np.asarray(vec).squeeze()
            # class-specific application mask:
            # rule applies if its pattern matches, its score is an improvement, AND
            # it matches the target class (positive for sign=1, negative for sign=0)
            if sign:
                apply_mask = (vec_dense == 1) & (score > trial_assign) & pos_mask
            else:
                apply_mask = (vec_dense == 1) & (score > trial_assign) & neg_mask
            
            trial_assign[apply_mask] = score

            # potential new global score
            current_global_score = np.mean(trial_assign)
            
            # gain relative to the current best global score
            gain = current_global_score - global_score

            if gain > best_gain: 
                improved = True
                best_gain = gain
                best_rule_name = name 
                best_assign = trial_assign 

        if best_rule_name is not None and best_gain > 0: 
            selected_rules.append(best_rule_name)
            current_assign = best_assign
            # update global_score with the gain from this iteration
            global_score = np.mean(current_assign) 
        else:
            break
    # uncovered occurrences
    num_pos_total = pos_mask.sum()
    num_neg_total = neg_mask.sum()
    uncovered_pos_occs = int(num_pos_total - (current_assign[pos_mask] > base_pos_prob).sum())
    uncovered_neg_occs = int(num_neg_total - (current_assign[neg_mask] > base_neg_prob).sum())

    return selected_rules, global_score, uncovered_pos_occs, uncovered_neg_occs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--npz", required=True, help="Path to the input npz file with matrices")
    parser.add_argument("--json", required=True, help="Json path with selected rules")
    parser.add_argument("--output", required=True, help="Path to the output with the name selection and eval")
    args = parser.parse_args()

    filename = args.npz.split("_data")[0]

    data = np.load(args.npz, allow_pickle=True)
    selected_indices = [pat['idx'] for pat in data['patterns']]
    selected_signs = [1 if pat['decision'] == 'yes' else 0 for pat in data['patterns']]
    selected_names = [pat['name'] for pat in data['patterns']]
    selected_precisions = [pat['precision'] for pat in data['patterns']]
    y = data['y']
    X = data['X'].item().toarray()
    X_selected = X[:, selected_indices]

    rules = [(name, X[:, i], score/100, sign)
        for i, sign, score, name in zip(selected_indices, selected_signs, selected_precisions, selected_names)
    ]

    selected_rules_max_precision, best_global_prec, uncovered_pos_occs, uncovered_neg_occs = greedy_forward_global_score_selection(
        X_selected,
        y, rules,
        rule_filter="all"
        )
    
    # Create a new X matrix with only the columns corresponding to selected_rules_max_precision
    selected_indices_max_precision = [selected_names.index(rule_name) for rule_name in selected_rules_max_precision]
    X_selected_max_precision = X_selected[:, selected_indices_max_precision]

    # Compute the measures on the new matrix
    global_coverage_score_new, uncovered_occs_new = global_coverage(X_selected_max_precision, y, [selected_signs[i] for i in selected_indices_max_precision])
    redudancy_score_new = redundancy(X_selected_max_precision)
    
    global_coverage_score, uncovered_occs = global_coverage(X_selected, y, selected_signs)
    redudancy_score = redundancy(X_selected)

    res = {
        "selected_rules_len": len(selected_rules_max_precision),
        "selected_rules": selected_rules_max_precision,
        "selected_indices": selected_indices_max_precision,
        "max_precision" : best_global_prec,
        "uncovered_pos_occs" : uncovered_pos_occs,
        "uncovered_neg_occs" : uncovered_neg_occs,
        "original_rules" : len(selected_indices),
        "global_coverage": global_coverage_score,
        "redudancy_score": redudancy_score,
    }

    with open(f"{filename}_greedy_description.json", "w") as f:
        json.dump(res, f)

    with open(args.json) as f:
        json_data = json.load(f)

    rules_to_keep = set(data['selected_rules'])
    filtered_rules = [rule for rule in json_data['rules'] if rule['pattern'] in rules_to_keep]

    json_data_copy = json_data.copy()
    json_data_copy['rules'] = filtered_rules

    with open(args.output, "w") as f_out:
        json.dump(json_data_copy, f_out, indent=2)
