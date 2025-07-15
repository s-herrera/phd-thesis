import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skglm
import argparse
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import sys
sys.path.insert(1, "..")
from utils import build_occurrences, matrix
from evaluation import global_coverage, max_global_precision, redundancy
from univariate import compute_rule_precision


def compute_global_metrics(X, y, feature_names):
    nonzero_cols = np.array((X != 0).sum(axis=0)).flatten() > 0
    X = X[:, nonzero_cols]
    feature_names = [feature_names[i] for i, keep in enumerate(nonzero_cols) if keep]
    rules = compute_rule_precision(X, y, feature_names)
    rule_signs = [tpl[1] for tpl in rules]
    print(X.shape, len(rule_signs), y.shape)
    global_cov = global_coverage(X, y, rule_signs)
    max_global_prec = max_global_precision(rules, y, rule_zero=True)
    red = redundancy(X)
    return max_global_prec, global_cov, red
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Learning Curve Analysis for Single Lasso Feature Selection")
    parser.add_argument("--alpha", type=float, default=0.001, help="")
    parser.add_argument("--path", required=True, help="Path to the input file")
    parser.add_argument("--patterns", required=True, help="Grex pattern file")
    args = parser.parse_args()

    data = build_occurrences(args.path, args.patterns, "sud")
    X_full, y_full, feature_names = matrix(data, max_degree=2, min_feature_occurence=5)

    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
    )
    print("Matrices done!")

    previous_selected_features_set = set()
    steps = 50
    subset_fractions = np.linspace(0.02, 1.0, steps).tolist()
    results = []

    print("--- Go subset evaluations ---")
    for fraction in subset_fractions:
        
        if fraction < 1.0:
            X_subset, _, y_subset, _ = train_test_split(
                X_train, y_train, train_size=fraction, random_state=42, stratify=y_train
            )
        else:
            X_subset, y_subset = X_train, y_train

        n_samples = X_subset.shape[0]
        print(f"\n--- Running pipeline with {n_samples} samples ({fraction*100:.0f}%) ---")
        
        model = skglm.SparseLogisticRegression(
            alpha=0.001,
            fit_intercept=True,
            max_iter=20,
            max_epochs=1000,
        )

        model.fit(X_subset, y_subset)
        
        y_test_pred = model.predict(X_test)
        
        test_f1 = f1_score(y_test, y_test_pred, average='macro')
        coefs = model.coef_.ravel()
        nonzero_idxs = np.nonzero(coefs)[0]
        num_selected_features = len(nonzero_idxs)
        selected_feature_names = np.array(feature_names)[nonzero_idxs]
        
        amp_scores, global_cov, red = compute_global_metrics(X_test[:, nonzero_idxs], y_test, selected_feature_names)
        current_selected_features_set = set(selected_feature_names)

        # jaccard similarity
        intersection = len(previous_selected_features_set.intersection(current_selected_features_set))
        union = len(previous_selected_features_set.union(current_selected_features_set))
        jaccard_stability = intersection / union if union > 0 else 0
        previous_selected_features_set = current_selected_features_set

        results.append({
            "num_samples": n_samples,
            "num_features_selected": num_selected_features,
            "macro_f1_score": test_f1,
            "jaccard_stability": jaccard_stability,
            "feature_names": selected_feature_names.tolist(),
            "amp": amp_scores[0],
            "global_cov": global_cov[0],
            "redundancy": red
        })


        


    if results:
        df_results = pd.DataFrame(results)
        df_results.to_json("data_amount_eval.json")
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(10, 12), sharex=True)

        ax1.plot(df_results['num_samples'], df_results['num_features_selected'])
        ax1.set_ylabel("Number of Selected Rules")
        ax1.set_title("")
        ax1.grid(True, linestyle='--', linewidth=0.5)

        ax2.plot(df_results['num_samples'], df_results['jaccard_stability'])
        ax2.set_ylabel("Jaccard Similarity to previous set")
        ax2.grid(True, linestyle='--', linewidth=0.5)
        ax2.set_ylim(0, 1)

        ax3.plot(df_results['num_samples'], df_results['macro_f1_score'])
        ax3.set_ylabel("Macro F1-Score on test set")
        ax3.grid(True, linestyle='--', linewidth=0.5)
        ax3.set_ylim(0, 1)

        ax4.plot(df_results['num_samples'], df_results['amp'])
        ax4.set_ylabel("AMP on test set")
        ax4.grid(True, linestyle='--', linewidth=0.5)
        ax4.set_ylim(0, 1)

        ax5.plot(df_results['num_samples'], df_results['global_cov'])
        ax5.set_ylabel("Global coverage of test set")
        ax5.set_xlabel("Number of Training Samples")
        ax5.grid(True, linestyle='--', linewidth=0.5)
        ax5.set_ylim(0, 1)

        plt.tight_layout()
        plt.show()