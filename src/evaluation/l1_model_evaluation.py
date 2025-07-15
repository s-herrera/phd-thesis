import json
import numpy as np
import skglm
import argparse

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score

from evaluation.global_eval_measures import global_coverage, average_maximum_precision, redundancy
from univariate.univariate import compute_rule_precision
from utils.matrix_utils import build_occurrences, matrix, align_matrix


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    maj_baseline_macro_f1 = majority_baseline_macro_f1(y_true)
    balanced_accuracy = np.nan
    unique_true_classes = np.unique(y_true)
    if len(unique_true_classes) > 1:
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

    prec_minority = np.nan
    recall_minority = np.nan

    if len(unique_true_classes) > 0:
        class_labels, counts = np.unique(y_true, return_counts=True)

        prec_per_class = precision_score(y_true, y_pred, average=None, labels=class_labels, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, labels=class_labels, zero_division=0)

        if len(class_labels) > 1:
            minority_class_idx = np.argmin(counts)
            print(minority_class_idx)
            prec_minority = prec_per_class[minority_class_idx]
            recall_minority = recall_per_class[minority_class_idx]
        elif len(class_labels) == 1:
            prec_minority = prec_per_class[0]
            recall_minority = recall_per_class[0]
    return acc, balanced_accuracy, prec_macro, recall_macro, f1_macro, maj_baseline_macro_f1, prec_minority, recall_minority,


def majority_baseline_macro_f1(y_true: np.ndarray) -> float:
    # TODO: I have to add also frequency majority baseline
    if np.mean(y_true) > 0.5:
        majority_class = 1
    else:
        majority_class = 0
        
    y_pred_baseline = np.full_like(y_true, fill_value=majority_class)
    macro_f1 = f1_score(y_true, y_pred_baseline, average='macro', zero_division=0)
    return macro_f1


def compute_global_metrics(X, y, feature_names):
    # X must contain only the features of interest (but because is aligned, has columns with only zeros).
    # Same for feature_names
    # rules = vec, sign, score

    # Drop columns in X that only have 0s
    nonzero_cols = np.array((X != 0).sum(axis=0)).flatten() > 0
    X = X[:, nonzero_cols]
    feature_names = [feature_names[i] for i, keep in enumerate(nonzero_cols) if keep]
    rules = compute_rule_precision(X, y, feature_names)
    rule_signs = [tpl[1] for tpl in rules]
    global_cov = global_coverage(X, y, rule_signs)
    amp = average_maximum_precision(rules, y, rule_zero=True)
    red = redundancy(X)
    return amp, global_cov, red


def eval_and_record(prefix, X, y, model, features, results, n_miss=0):

    y_pred = model.predict(X)
    majority_baseline = max(y.mean(), 1 - y.mean())
    metrics = compute_metrics(y, y_pred)
    amp_scores, cov, red = compute_global_metrics(X, y, features)

    accuracy, bal_accuracy, prec_macro, recall_macro, f1_macro, maj_f1_macro, prec_minority, recall_minority = metrics
    f1_gain = f1_macro - maj_f1_macro
    amp, amp_pos, amp_neg, amp_zero_rule, (base_pos, base_neg) = amp_scores
    coverage, (not_covered_total, not_covered_pos, not_covered_neg, not_covered_indices) = cov

    results[prefix] = {
        "accuracy": accuracy,
        "balanced_accuracy": bal_accuracy,
        "precision_macro": prec_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "f1_gain": f1_gain,
        "maj_f1_macro_baseline": maj_f1_macro,
        "precision_minority": prec_minority,
        "recall_minority": recall_minority,
        "majority_class_baseline": majority_baseline,
        "amp": amp,
        "pos_amp": amp_pos,
        "neg_amp": amp_neg,
        "rule_zero_amp": amp_zero_rule,
        "pos_base_precision": base_pos,
        "neg_base_precision": base_neg,
        "global_coverage": coverage,
        "not_covered_total": not_covered_total,
        "not_covered_pos": not_covered_pos,
        "not_covered_neg": not_covered_neg,
        "redundancy": red,
        "rules_missed": n_miss
    }

# TODO: factorize better the script
def eval_train_test(path, output, patterns, corpora=None, max_degree=2, grew_config="sud"):
    results = dict()
    data = build_occurrences(path, patterns, grew_config)
    X_full, y_full, feature_names = matrix(data, max_degree, min_feature_occurence=5)
    X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.2, random_state=42, stratify=y_full)
    print("Matrices done!")

    model_A = skglm.SparseLogisticRegression(
        alpha=0.001,
        fit_intercept=True,
        max_iter=20,
        max_epochs=1000,
    )

    # fit the model to select features
    model_A.fit(X_train, y_train)
    selector = SelectFromModel(model_A, prefit=True)

    X_train_selected = selector.transform(X_train)
    X_test_selected  = selector.transform(X_test)

    keep_mask = selector.get_support()
    selected_feature_names = [
        fname for fname, keep in zip(feature_names, keep_mask) if keep
    ]

    # Model B with the selected features, a "dense" model. This evaluation is then equal to the evaluation of selected features in other datasets.
    # Do I need to balance the dataset? I think so, it's a better evaluation.
    model_B = LogisticRegression(
        penalty='l2', 
        fit_intercept=True, 
        max_iter=1000, 
        random_state=42,
        class_weight='balanced'
    )
    # train now only with selected features
    model_B.fit(X_train_selected, y_train)

    eval_and_record(path, X_train_selected, y_train, model_B, selected_feature_names, results)
    eval_and_record(f"{path}_test", X_test_selected, y_test, model_B, selected_feature_names, results)

    if corpora:
        for corpus in corpora:

            data_test = build_occurrences(corpus, patterns, grew_config)

            X_test_corpus, y_test_corpus, feature_names_corpus = matrix(data_test, max_degree=2, min_feature_occurence=5)
            X_test_corpus_aligned, test_corpus_missing_features = align_matrix(
                X_full,
                X_test_corpus,
                feature_names,
                feature_names_corpus
            )

            X_selected_test_corpus = selector.transform(X_test_corpus_aligned)

            nz_per_col = X_selected_test_corpus.getnnz(axis=0)
            present_mask = nz_per_col > 0
            n_sel = X_selected_test_corpus.shape[1]
            n_pres = int(present_mask.sum())
            n_miss = n_sel - n_pres
            print("Original Selected Rules", n_sel)
            print("Missed", n_miss)
            print(f"\nEvaluating on external corpus: {corpus} ({X_selected_test_corpus.shape[0]} samples)")
            eval_and_record(f"{corpus}_test", X_selected_test_corpus, y_test_corpus, model_B, selected_feature_names, results, n_miss)


    with open(output, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--path", required=True, help="Path to the input file")
    parser.add_argument("--output", required=True, help="Path to the output file")
    parser.add_argument("--patterns", required=True, help="Grex pattern file")
    parser.add_argument("--max-degree", default="2", type=int)
    parser.add_argument("--config", default="sud")
    parser.add_argument("--test-corpora", nargs="+", required=False, help="List of files to compare to")

    args = parser.parse_args()
    if args.test_corpora:
        eval_train_test(args.path, args.output, args.patterns, args.test_corpora, max_degree=args.max_degree, grew_config=args.config)
    else:
        eval_train_test(args.path, args.output, args.patterns, max_degree=args.max_degree, grew_config=args.config)