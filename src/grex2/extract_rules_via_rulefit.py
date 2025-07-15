import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from utils import build_occurrences, matrix
import scipy
import argparse
from scipy import sparse
import numpy as np
import skglm
import json
import scipy


def train_gradient_boosting(X, y, n_estimators=10, max_depth=3, learning_rate=0.1, random_state=42):
    gbt = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        subsample=1,
        min_samples_leaf=5,
    )
    gbt.fit(X, y)
    return gbt


def extract_rules_from_gbm(gbt_model, feature_names, max_conditions):
    """
    Extracts rules for RuleFit.
    A rule is any path with length > 1 and <= max_conditions.
    """
    extracted_rules_conditions = []
    extracted_rule_names = []
    unique_rule_strings = set()

    for tree_idx, base_estimator_array in enumerate(gbt_model.estimators_):
        decision_tree = base_estimator_array[0]
        tree_ = decision_tree.tree_

        # rulefit
        def find_paths(node_id, current_conditions, current_rule_str_parts, depth):
            # depth is the number of conditions in current_conditions / current_rule_str_parts
            if depth > 1 and depth <= max_conditions:
                if current_rule_str_parts:  # should always be true if depth > 1
                    rule_str = ",".join(current_rule_str_parts)
                    if rule_str not in unique_rule_strings:
                        unique_rule_strings.add(rule_str)
                        extracted_rules_conditions.append(list(current_conditions))
                        extracted_rule_names.append(rule_str)

            is_leaf = tree_.feature[node_id] == -2
            if is_leaf or depth >= max_conditions:
                return

            # if not a leaf and depth < max_conditions, this is a split node.
            feature_idx = tree_.feature[node_id]
            threshold = tree_.threshold[node_id]

            if feature_idx < 0 or feature_idx >= len(feature_names):
                print(
                    f"Warning: Invalid feature_idx {feature_idx} encountered in tree {tree_idx}, node {node_id}."
                )
                return

            feature_name = feature_names[feature_idx]

            # recurse for left child
            left_condition = (feature_idx, "<=", threshold)
            left_condition_str = f"0:{feature_name}"
            find_paths(
                tree_.children_left[node_id],
                current_conditions + [left_condition],
                current_rule_str_parts + [left_condition_str],
                depth + 1,
            )

            # recurse for right child
            right_condition = (feature_idx, ">", threshold)
            right_condition_str = f"1:{feature_name}"
            find_paths(
                tree_.children_right[node_id],
                current_conditions + [right_condition],
                current_rule_str_parts + [right_condition_str],
                depth + 1,
            )

        find_paths(node_id=0, current_conditions=[], current_rule_str_parts=[], depth=0)
    return extracted_rules_conditions, extracted_rule_names


def apply_rules_to_data(X_input, rules_conditions_list):
    """
    From original X data, I apply the selected decision rules to get resulting matrix.
    """

    # TODO: I tranform this in a dense matrix, it is not optimal.
    if scipy.sparse.issparse(X_input):
        print(
            "X_input is sparse. Converting to dense array."
        )
        X_dense = X_input.toarray()

    elif not isinstance(X_input, np.ndarray):
        try:
            X_dense = np.array(X_input, copy=False)
        except Exception as e:
            raise TypeError(f"X_input could not be converted to a NumPy array: {e}")
    else:
        X_dense = X_input

    # check if have the good dimentions
    if X_dense.ndim == 0: 
        X_dense = X_dense.reshape(1, 1)
    elif X_dense.ndim == 1:  # 1D array
        X_dense = X_dense.reshape(1, -1)

    if X_dense.ndim != 2:
        raise ValueError(
            f"X_input must be effectively 2D. Got {X_dense.ndim} dimensions after processing."
        )
    

    n_samples = X_dense.shape[0]
    n_features_in_X = X_dense.shape[1]

    if not rules_conditions_list:
        return np.array([]).reshape(n_samples, 0)

    n_rules = len(rules_conditions_list)
    X_rules = np.zeros((n_samples, n_rules), dtype=np.int8)

    for rule_idx, conditions_for_one_rule in enumerate(rules_conditions_list):

        condition_evaluations_for_all_samples = []
        current_rule_is_valid = True

        # evaluate each condition in the rule for all samples
        # threshold_val is always 0.5, but it can be generalize to continuous values.
        for feature_idx, operator_str, threshold_val in conditions_for_one_rule:
            if feature_idx < 0 or feature_idx >= n_features_in_X:
                print(
                    f"Warning: Rule {rule_idx} contains feature_idx {feature_idx},"
                    f"which is out of bounds for input X with {n_features_in_X} features."
                    f"This rule will evaluate to False for all samples."
                )
                current_rule_is_valid = False
                break  # stop processing this rule

            # get the entire column
            feature_column = X_dense[:, feature_idx]

            # perform comparison for this condition
            if operator_str == "<=":
                eval_for_this_condition = feature_column <= threshold_val
            elif operator_str == ">":
                eval_for_this_condition = feature_column > threshold_val
            else:
                print(
                    f"Warning: Unknown operator '{operator_str}' in Rule {rule_idx}. "
                    "This condition will evaluate as False for all samples."
                )
                eval_for_this_condition = np.zeros(n_samples, dtype=bool)
            condition_evaluations_for_all_samples.append(eval_for_this_condition)

        if not current_rule_is_valid:
            continue

        # if the rule is valid and had conditions, combine the condition evaluations
        if condition_evaluations_for_all_samples:
            # for a rule to be true, all its conditions must be true (AND logic)
            # np.logical_and.reduce performs an element-wise AND across a list of boolean arrays
            final_rule_evaluation = np.logical_and.reduce(
                condition_evaluations_for_all_samples
            )
            X_rules[:, rule_idx] = final_rule_evaluation.astype(np.int8)
    return X_rules


def filter_rules(initial_rules_conditions, initial_rule_names):
    filtered_rules_conditions = []
    filtered_rule_names = []

    if initial_rules_conditions:  # only filter if there are rules to filter
        for i, rule_conditions_list_for_one_rule in enumerate(initial_rules_conditions):
            # I only get those that have at least one positive condition.
            has_at_least_one_presence_condition = False
            for (_, operator_str, _) in rule_conditions_list_for_one_rule:
                # Assuming features 0 or 1 and threshold around 0.5:
                # operator_str == ">" implies feature == 1 (presence of the linguistic property)
                if operator_str == ">":
                    has_at_least_one_presence_condition = True
                    break  # this rule is good

            if has_at_least_one_presence_condition:
                filtered_rules_conditions.append(rule_conditions_list_for_one_rule)
                filtered_rule_names.append(initial_rule_names[i])

        print(f"After filtering, {len(filtered_rule_names)} rules remain.\n")

        # update the variables to use the filtered lists for subsequent steps
        rules_conditions = filtered_rules_conditions
        rule_names = filtered_rule_names
    else:
        # no initial rules to filter, so keep them as empty
        rules_conditions = initial_rules_conditions
        rule_names = initial_rule_names
    return rules_conditions, rule_names


if __name__ == "__main__":
    cmd = argparse.ArgumentParser(description="Train Gradient Boosting and extract rules.")
    cmd.add_argument("data", metavar="F", type=str, nargs="+", help="data")
    cmd.add_argument("--patterns",type=str,required=True, help="Path to the patterns file for build_occurrences.")
    cmd.add_argument("--config", type=str, default="ud", help="Path to the GREW configuration file.")
    cmd.add_argument("--output", type=str, required=True, help="Output JSON file to save extracted rules.")
    cmd.add_argument("--n_estimators", type=int, default=500, help="Number of estimators for Gradient Boosting.",)
    cmd.add_argument("--npz", action="store_true")
    args = cmd.parse_args()

    data = build_occurrences(args.data, args.patterns, args.config)
    X_initial, y, feature_names = matrix(data, max_degree=1)

    N_ESTIMATORS = args.n_estimators
    MAX_RULE_CONDITIONS = 3
    MAX_DEPTH = 5
    RANDOM_STATE = 42

    rules_conditions = []
    rule_names = []

    X_train, _, y_train, _ = train_test_split(X_initial, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    print(f"Training GBT with {N_ESTIMATORS} estimators...")
    gbt_model = train_gradient_boosting(
        X_train,
        y_train,
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
    )
    print("Gradient Boosting model trained.\n")

    print(f"Extracting rules with exactly max {MAX_RULE_CONDITIONS} conditions...")
    initial_rules_conditions, initial_rule_names = extract_rules_from_gbm(gbt_model, feature_names, MAX_RULE_CONDITIONS)
    rules_conditions, rule_names = filter_rules(initial_rules_conditions, initial_rule_names)

    # X_rule_features = apply_rules_to_data(X, rules_conditions)
    X_rule_features = apply_rules_to_data(X_initial, rules_conditions)
    if not rules_conditions:
        print("No rules were extracted or training was skipped.")
        X_final = X_initial
        all_feature_names = list(feature_names)
    else:
        print(f"Extracted {len(rule_names)} unique rules.\n")
        print(f"Shape of original X (ensured 2D for hstack): {X_initial.shape}")
        print(f"Shape of new rule-based features: {X_rule_features.shape}\n")

        X_final = sparse.hstack((X_initial, X_rule_features))
        all_feature_names = list(feature_names) + rule_names
        print(f"Shape of final combined X: {X_final.shape}")
        print(f"Total number of features after adding rules: {len(all_feature_names)}\n")

    print(f"Final X data shape: {X_final.shape}")
    num_original_feats = X_initial.shape[1]
    print(f"Number of original features: {num_original_feats}")
    print(f"Number of new rule features: {X_final.shape[1] - num_original_feats}")
    print(f"Total features: {len(all_feature_names)}")

    col_counts = np.array((X_final != 0).sum(axis=0)).flatten()
    keep_cols = np.where(col_counts >= 5)[0]
    if sparse.issparse(X_final):
        X_final = X_final.tocsc()[:, keep_cols]
    else:
        X_final = X_final[:, keep_cols]
    print("with threshold", X_final.shape)
    all_feature_names = [all_feature_names[i] for i in keep_cols]

    alpha_start = 0.0001
    alpha_end = 0.0001
    alpha_num = 1
    num_positive = np.sum(y)
    output = args.output
    X = X_final.tocsr()

    extracted_rules = dict()
    extracted_rules["data_len"] = len(data)
    extracted_rules["n_yes"] = int(num_positive)
    extracted_rules["intercepts"] = list()

    classification_data = {"X": X, "y": y, "patterns": list()}

    # extract rules
    all_rules = set()
    ordered_rules = list()

    alphas = alphas = np.linspace(alpha_start, alpha_end, alpha_num)
    for j, alpha in enumerate(alphas):
        print("extracting rules (%i / %i)" % (j + 1, len(alphas)), flush=True)
        model = skglm.SparseLogisticRegression(
            alpha=alpha,
            fit_intercept=True,
            max_iter=20,
            max_epochs=1000,
        )
        model.fit(X, y)
        extracted_rules["intercepts"].append((alpha, model.intercept_))

        for idx, value in enumerate(model.coef_[0]):
            if value == 0:
                continue
            name = all_feature_names[idx]
            if name not in all_rules:
                all_rules.add(name)
                col = np.asarray(X[:, idx].todense())
                idx_col = col.squeeze(1)

                with_feature_selector = idx_col > 0
                without_feature_selector = np.logical_not(with_feature_selector)

                matched = y[with_feature_selector]
                n_matched = len(matched)
                n_pattern_positive_occurence = matched.sum()
                n_pattern_negative_occurence = n_matched - n_pattern_positive_occurence

                mu = num_positive / len(data)
                a = n_pattern_positive_occurence / n_matched
                gstat = (
                    2
                    * n_matched
                    * (
                        ((a * np.log(a)) if a > 0 else 0)
                        - a * np.log(mu)
                        + (((1 - a) * np.log(1 - a)) if (1 - a) > 0 else 0)
                        - (1 - a) * np.log(1 - mu)
                    )
                )
                p_value = 1 - scipy.stats.chi2.cdf(gstat, 1)
                cramers_phi = np.sqrt((gstat / n_matched))

                expected = (n_matched * num_positive) / len(data)
                delta_observed_expected = n_pattern_positive_occurence - expected

                if n_pattern_positive_occurence / n_matched > int(y.sum()) / len(data):
                    decision = "yes"
                    coverage = (n_pattern_positive_occurence / num_positive) * 100
                    precision = (n_pattern_positive_occurence / n_matched) * 100
                else:
                    decision = "no"
                    coverage = (
                        n_pattern_negative_occurence / (len(data) - num_positive)
                    ) * 100
                    precision = (n_pattern_negative_occurence / n_matched) * 100

                ordered_rules.append(
                    {
                        "pattern": name,
                        "n_pattern_occurence": int(idx_col.sum()),
                        "n_pattern_positive_occurence": int(n_pattern_positive_occurence),
                        "decision": decision,
                        "alpha": alpha,
                        "value": value,
                        "coverage": coverage,
                        "precision": precision,
                        "delta": delta_observed_expected,
                        "g-statistic": gstat,
                        "p-value": p_value,
                        "cramers_phi": cramers_phi,
                    }
                )

                classification_data["patterns"].append(
                    {
                        "name": name,
                        "idx": idx,
                        "decision": decision,
                        "alpha": alpha,
                        "precision": precision,
                        "coverage": coverage,
                    }
                )

    extracted_rules["rules"] = ordered_rules

    print("Done.", flush=True)
    with open(output, "w") as out_stream:
        json.dump(extracted_rules, out_stream)

    if args.npz:
        np.savez(
            args.output.split(".json")[0] + "_data",
            X=classification_data["X"],
            y=classification_data["y"],
            patterns=classification_data["patterns"],
        )
