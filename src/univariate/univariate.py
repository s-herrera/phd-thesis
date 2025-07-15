from sklearn.metrics import precision_score, recall_score
from sklearn.feature_selection import chi2, mutual_info_classif
import numpy as np

def univariate_measures(X, y, feature_names):
    results = {}
    # Compute chi2 and mutual info for all features at once
    chi2_scores, chi2_pvalues = chi2(X, y)
    mi_scores = mutual_info_classif(X, y, discrete_features='auto')
    base_pos_prob = np.mean(y)

    for idx, feature in enumerate(feature_names):
        feature_results = {}
        X_col = X[:, idx]
        counts = np.sum(X_col)

        if hasattr(X_col, "toarray"):
            X_col = X_col.toarray().ravel()
        else:
            X_col = np.asarray(X_col).ravel()

        if np.sum(X_col) > 0:
            p_y1_given_feature = np.mean(y[X_col == 1])
            predicted_class = p_y1_given_feature > base_pos_prob
        else:
            feature_results['p_y1_given_feature'] = None
            feature_results['predicted_class'] = None

        if predicted_class:
            precision = precision_score(y, X_col, pos_label=1, zero_division=0)
            cov = recall_score(y, X_col, pos_label=1, zero_division=0)
        else:
            neg_X_col = np.logical_not(X_col)
            precision = precision_score(y, neg_X_col, pos_label=0, zero_division=0)
            cov = recall_score(y, neg_X_col, pos_label=0, zero_division=0)

        feature_results['idx'] = idx
        feature_results['occs'] = counts
        feature_results['precision'] = precision
        feature_results['coverage'] = cov
        feature_results['mutual_info'] = mi_scores[idx]
        feature_results['chi2'] = chi2_scores[idx]
        feature_results['p-value'] = chi2_pvalues[idx]
        feature_results['p_y1_given_feature'] = p_y1_given_feature
        feature_results['predicted_class'] = predicted_class
        results[feature] = feature_results

    return results


def compute_rule_precision(X, y, feature_names):
    """return tuples with (feature vector, feature sign, feature precision)"""
    results = []
    base_pos_prob = np.mean(y)
    for idx, feature in enumerate(feature_names):
        feature_results = {}
        X_col = X[:, idx]

        if hasattr(X_col, "toarray"):
            X_col = X_col.toarray().ravel()
        else:
            X_col = np.asarray(X_col).ravel()

        if np.sum(X_col) > 0:
            p_y1_given_feature = np.mean(y[X_col == 1])
            predicted_class = p_y1_given_feature > base_pos_prob
        else:
            feature_results['p_y1_given_feature'] = None
            feature_results['predicted_class'] = None

        if predicted_class:
            precision = precision_score(y, X_col, pos_label=1, zero_division=0)
        else:
            neg_X_col = np.logical_not(X_col)
            precision = precision_score(y, neg_X_col, pos_label=0, zero_division=0)

        results.append((X_col, 1 if predicted_class else 0, precision))

    return results


def top_percentile_rules(results, score, percentile=20):
    scores = np.array([v[score] for v in results.values()])
    threshold = np.percentile(scores, 100 - percentile)
    top_features = {k: v for k, v in results.items() if v[score] >= threshold}
    return top_features