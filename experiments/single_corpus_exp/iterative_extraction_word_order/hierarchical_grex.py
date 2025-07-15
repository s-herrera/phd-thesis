import numpy as np
from collections import deque
from typing import List
from utils import matrix, build_occurrences
import pandas as pd
import matplotlib.pyplot as plt
import skglm
import argparse
from scipy.sparse import hstack
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)

sys.path.insert(1, "../")
from evaluation import global_coverage, max_global_precision, redundancy
from univariate import compute_rule_precision
import time

def compute_global_metrics(X, y, feature_names):
    nonzero_cols = np.array((X != 0).sum(axis=0)).flatten() > 0
    X = X[:, nonzero_cols]
    feature_names = [feature_names[i] for i, keep in enumerate(nonzero_cols) if keep]
    rules = compute_rule_precision(X, y, feature_names)
    rule_signs = [tpl[1] for tpl in rules]
    global_cov = global_coverage(X, y, rule_signs)
    max_global_prec = max_global_precision(rules, y, rule_zero=True)
    red = redundancy(X)
    return max_global_prec, global_cov, red
    

def find_hierarchical_rules(X, y, feature_names: list, config: dict) -> list:
    """
    Finds hierarchical rules by iteratively running an L1-regularized model
    on progressively smaller subsets of the data, removing redundant rules
    and limiting to top-K features per node based on coefficient magnitude.
    """
    all_rules = []
    seen_supports = set()
    remaining = deque()
    rule_id_counter = 0

    # Initialize with full dataset indices
    initial_indices = np.arange(X.shape[0])
    remaining.append(
        {"parent_rule_id": None, "data_indices": initial_indices, "current_depth": 0}
    )

    total_pos_initial = int(y.sum())
    total_neg_initial = len(y) - total_pos_initial

    while remaining:
        current_scope = remaining.popleft()
        depth = current_scope["current_depth"]
        indices = current_scope["data_indices"]
        n_samples = len(indices)

        # Stop if max depth or too few samples
        if depth >= config.get("max_depth", 1) or n_samples < config.get(
            "min_samples", 1
        ):
            continue

        # Subset data
        X_scope = X[indices, :]
        y_scope = y[indices]
        total_pos = int(y_scope.sum())
        total_neg = n_samples - total_pos

        Xsub = X_scope.tocsc()
        min_s = config.get("min_samples", 1)
        supports = Xsub.getnnz(axis=0)
        valid_mask = supports >= min_s
        if not valid_mask.any():
            continue

        X_subset = Xsub[:, valid_mask]
        feat_map = np.nonzero(valid_mask)[0]

        # L1 logistic model
        model = skglm.SparseLogisticRegression(
            alpha=config.get("alpha", 0.001), fit_intercept=True
        )
        try:
            model.fit(X_subset, y_scope)
        except Exception as e:
            print(f"Warning: model fit failed at depth {depth}: {e}")
            continue

        # Select top-K features by absolute coefficient
        coefs = model.coef_.ravel()
        nonzero_idxs = np.nonzero(coefs)[0]
        if nonzero_idxs.size == 0:
            continue
        # rank by absolute coefficient
        ranks = nonzero_idxs[np.argsort(np.abs(coefs[nonzero_idxs]))[::-1]]
        k = config.get("top_k", len(ranks))
        selected_idxs = ranks[:k]

        # Generate rules for selected features
        for rel_idx in selected_idxs:
            coef = coefs[rel_idx]
            orig_idx = int(feat_map[rel_idx])
            feat_name = feature_names[orig_idx]

            # support rows within scope
            col = X_subset.getcol(rel_idx)
            support_rows = col.nonzero()[0]
            p_counts = support_rows.size
            if p_counts < min_s:
                continue

            # absolute support indices
            abs_support = indices[support_rows]
            support_key = frozenset(int(i) for i in abs_support)
            # skip redundant rules
            if support_key in seen_supports:
                continue
            seen_supports.add(support_key)

            # positive/negative counts
            matched = y_scope[support_rows]
            p_q = int(matched.sum())
            not_p_q = p_counts - p_q

            # local vs global precision
            local_prec = p_q / p_counts
            global_prec = total_pos / n_samples if n_samples > 0 else 0

            if local_prec > global_prec:
                if p_q < min_s:
                    continue
                decision = "yes"
                coverage = (p_q / total_pos) * 100 if total_pos else 0
                precision = local_prec * 100
                global_coverage = (
                    (p_q / total_pos_initial) * 100 if total_pos_initial else 0
                )
            else:
                if not_p_q < min_s:
                    continue
                decision = "no"
                coverage = (not_p_q / total_neg) * 100 if total_neg else 0
                precision = (not_p_q / p_counts) * 100
                global_coverage = (
                    (not_p_q / total_neg_initial) * 100 if total_neg_initial else 0
                )

            # rule
            new_rule = {
                "id": rule_id_counter,
                "parent_rule_id": current_scope["parent_rule_id"],
                "pattern": feat_name,
                "p_counts": p_counts,
                "p_q_counts": p_q,
                "not_p_q_counts": not_p_q,
                "decision": decision,
                "coef": float(coef),
                "coverage": coverage,
                "global_coverage": global_coverage,
                "precision": precision,
                "original_feature_index": orig_idx,
            }
            all_rules.append(new_rule)
            rule_id_counter += 1

            # child scope
            if 0 < p_counts < n_samples:
                child_indices = indices[support_rows]
                remaining.append(
                    {
                        "parent_rule_id": new_rule["id"],
                        "data_indices": child_indices,
                        "current_depth": depth + 1,
                    }
                )

    print(f"Total rules discovered: {rule_id_counter}")
    return all_rules


def print_rule_hierarchy(
    rules: List[dict], parent_id: int = None, indent: str = "", file=None
):
    """Recursively prints the discovered rules (stored as dicts) to show the hierarchy.
    If file is provided, also writes the output to the file.
    """
    children = [rule for rule in rules if rule["parent_rule_id"] == parent_id]
    for child in sorted(children, key=lambda r: r["precision"], reverse=True):
        line = f"{indent}{child['pattern']}, P counts: {child['p_counts']}, Prec%: {child['precision']:.1f}%, Cov%: {child['coverage']:.1f}%,  Order: {'left' if child['decision'] == 'no' else 'right'}"

        if file is not None:
            file.write(line + "\n")
        print_rule_hierarchy(
            rules, parent_id=child["id"], indent=indent + "  ", file=file
        )


def get_feature_path(rule_id, rule_map):
    """
    Return the list of original_feature_index values
    from the root down to this rule.
    """
    path = []
    while rule_id is not None:
        r = rule_map[rule_id]
        path.append(r["original_feature_index"])
        rule_id = r["parent_rule_id"]
    return list(reversed(path))  # root first


def build_rule_matrix(X, rules, rule_map):
    """
    Given a sparse X (n_samples × n_atomic_features)
    and a list of rules (with .id and .parent_rule_id),
    return X_rules (n_samples × n_rules) where each column
    is the elementwise AND of the atomic feature columns along
    that rule's path.
    """
    cols = []
    for r in rules:
        idx_path = get_feature_path(r["id"], rule_map)
        # start with the first column
        col = X[:, idx_path[0]].copy().tocsc()
        # successive multiplications
        for fi in idx_path[1:]:
            col = col.multiply(X[:, fi])
        cols.append(col)
    return hstack(cols, format="csc")


def predict_with_rule_set(X_test, selected_rules, training_y):
    """
    Makes predictions on test data using the 'highest precision wins' logic.

    Args:
        X_test: The feature matrix of the test data.
        selected_rules: The final list of rules, each a tuple of
                        (vec, sign, score).
        training_y: The original y_train vector, to calculate the default class.

    Returns:
        An array of final predictions (0 or 1).
    """
    n_occs = X_test.shape[0]
    majority_class = 1 if np.mean(training_y) > 0.5 else 0
    predictions = np.full(n_occs, majority_class, dtype=int)
    best_scores = np.zeros(n_occs, dtype=float)
    for _, vec_template, score, sign in selected_rules:
        vec_dense = np.asarray(vec_template.todense()).squeeze()
        apply_mask = (vec_dense == 1) & (score > best_scores)
        best_scores[apply_mask] = score
        predictions[apply_mask] = 1 if sign else 0
    return predictions


def rules_for_amp(rules, X):
    rules_for_amp = []
    for idx, rule in enumerate(rules):
        col_sparse = X[:, idx]
        col_array = col_sparse.toarray().ravel()
        rule_sing = 0 if rule["decision"] == "no" else 1
        rules_for_amp.append((col_array, rule_sing, rule["precision"]))
    return rules_for_amp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("data", metavar="F", type=str, nargs="+", help="data")
    parser.add_argument("--patterns", type=str, required=True)
    parser.add_argument("--config", type=str, default="ud")
    parser.add_argument("--max-degree", type=int, default=1)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--min-feature_occurence", type=int, default=5)
    parser.add_argument("--alpha-end", type=float, default=0.001)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()
    start_time = time.time()

    config = {
        "alpha": args.alpha_end,
        "max_depth": args.max_depth,
        "min_samples": args.min_feature_occurence,
        "top_k": args.top_k,
    }

    data = build_occurrences(args.data, args.patterns, args.config)
    print("Data loaded!")

    X_full, y_full, feature_names = matrix(
        data,
        max_degree=args.max_degree,
        min_feature_occurence=args.min_feature_occurence,
    )
    print("Matrices Done!")

    if args.eval:
        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
        )
        print(
            f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}"
        )

        rules = find_hierarchical_rules(X_train, y_train, feature_names, config)
        if not rules:
            raise SystemExit(
                "No rules were discovered on the training data. Evaluation cannot proceed."
            )
        rule_map = {r["id"]: r for r in rules}

        # matrices for eval
        X_train_rules = build_rule_matrix(X_train, rules, rule_map)
        X_test_rules = build_rule_matrix(X_test, rules, rule_map)

        rule_sings = [0 if r["decision"] == "no" else 1 for r in rules]
        train_global_cov, train_occs_not_covered = global_coverage(
            X_train_rules, y_train, rule_sings
        )

        test_global_cov, test_occs_not_covered = global_coverage(
            X_test_rules, y_test, rule_sings
        )

        train_rules_for_amp = rules_for_amp(rules, X_train_rules)
        test_rules_for_amp = rules_for_amp(rules, X_test_rules)
        train_amp = max_global_precision(train_rules_for_amp, y_train)
        test_amp = max_global_precision(test_rules_for_amp, y_test)

        for idx in test_occs_not_covered[-1]:
            print(data[idx])

        print(f"Built rule matrix with shape {X_train_rules.shape}")

        clf = LogisticRegression(
            penalty="l2",
            max_iter=1000,
            random_state=42,
        )

        clf.fit(X_train_rules, y_train)
        y_pred = clf.predict(X_test_rules)

        cv_scores = cross_val_score(
            clf, X_train_rules, y_train, cv=5, scoring="balanced_accuracy"
        )
        print(
            f"5-fold CV on rule‐features: mean={cv_scores.mean():.3f}, std={cv_scores.std():.3f}"
        )

        print(f"Accuracy of test: {accuracy_score(y_test, y_pred):.3f}")
        print(classification_report(y_test, y_pred))
        
    else:
        rules = find_hierarchical_rules(
            X_full, y_full, feature_names, config
        )

        if not rules:
            print("No rules were discovered.")
        else:
            with open("hierarchical_rules.txt", "w") as f:
                print_rule_hierarchy(rules, file=f)


        rule_map = {r["id"]: r for r in rules}
        # matrices for eval
        X_train_rules = build_rule_matrix(X_full, rules, rule_map)
        rule_sings = [0 if r["decision"] == "no" else 1 for r in rules]
        global_cov, occs_not_covered = global_coverage(X_train_rules, y_full, rule_sings)

        rules_for_amp = rules_for_amp(rules, X_train_rules)
        amp = max_global_precision(rules_for_amp, y_full)

        print("global_coverage:", global_cov)
        print("amp:", amp[0])

    stop_time = time.time()
    print("Time:", (stop_time - start_time) / 60)
