import numpy as np

def global_coverage(X, y, rule_signs):
    """
    X is the binary matrix with the selected features.
    y is the target variable vector.
    rule_signs indicates if the rule/feature predicts more 1 or 0.

    It computes global coverage, that is, if features cover the occurrences given the value of y and of the sign.
    It returns:
        coverage: fraction of covered occurrences
        n_not_covered: number of occurrences not covered
    """
    n_occs = X.shape[0]
    n_features = X.shape[1]

    covered = np.zeros(n_occs, dtype=bool)
    for i in range(n_features):
        col = X[:, i]
        if hasattr(col, "toarray"):
            col = col.toarray().ravel()
        else:
            col = np.asarray(col).ravel()

        if rule_signs[i] == 1:
            mask = (col == 1) & (y == 1)
        else:
            mask = (col == 1) & (y == 0)
        covered = np.logical_or(covered, mask)  # covered | mask
    coverage = covered.sum() / n_occs
    n_not_covered = np.sum(np.logical_not(covered))
    n_pos_not_covered = np.sum(np.logical_not(covered)[y == 1])
    n_neg_not_covered = n_not_covered - n_pos_not_covered
    not_covered_indices = np.where(np.logical_not(covered))[0]

    return coverage, (
        int(n_not_covered),
        int(n_pos_not_covered),
        int(n_neg_not_covered),
        not_covered_indices,
    )


def average_maximum_precision(rules, y, rule_zero=True):
    """
    How well each instance is covered by the best rule of its own class type.
    rules have vec, sign, score
    sign = 1/0
    if rule_zero, computes the base distribution given a scope and a conclusion
    It returns:
        amp: the mean of the max precision for each occurrence given the class type (decision=yes/no)
        pos_amp: the mean of the max precision for each positive occurrence
        neg_amp: the mean of the max precision for each negative occurrence
        rule_zero_amp: the initial mean, including or not the distribution of the rule zero.
        (pos_base_precision, neg_base_precision): the base distributions for y==1 and y==0
    """
    n_occs = len(y)
    max_precisions = np.zeros(n_occs, dtype=float)

    pos_base_precision = np.sum(y) / n_occs
    neg_base_precision = 1 - pos_base_precision

    rule_zero_amp = np.mean(
        np.where(y == 1, pos_base_precision, neg_base_precision)
    )

    if rule_zero:
        max_precisions[y == 1] = pos_base_precision
        max_precisions[y == 0] = neg_base_precision

    for vec, sign, score in rules:
        assert sign in [0, 1]

        if hasattr(vec, "toarray"):
            vec = vec.toarray().ravel()
        else:
            vec = np.asarray(vec).ravel()

        if sign == 1:
            match_mask = (vec == 1) & (y == 1)
        else:
            match_mask = (vec == 1) & (y == 0)
        max_precisions[match_mask] = np.maximum(max_precisions[match_mask], score)

    amp = np.mean(max_precisions)
    pos_amp = np.mean(max_precisions[y == 1])
    neg_amp = np.mean(max_precisions[y == 0])
    return (
        amp,
        pos_amp,
        neg_amp,
        rule_zero_amp,
        (pos_base_precision, neg_base_precision),
    )


def redundancy(X):
    """
    Calculates the average number of times an occurrence is covered by a rule.
    Returns:
        redundancy
    """
    coverage_counts = np.sum(X == 1, axis=1)
    redundancy_score = np.mean(coverage_counts)
    return redundancy_score


def redundancy_jaccard(X):
    """
    Computes the average pairwise Jaccard similarity between all columns (features) of X.
    This measures redundancy: higher average Jaccard means more overlap (redundancy) between features.
    Returns:
        mean_jaccard: average Jaccard similarity between all pairs of features
    """
    n_features = X.shape[1]
    if n_features < 2:
        return 0.0
    jaccard_scores = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            col_i = X[:, i]
            col_j = X[:, j]

            if hasattr(col_i, "toarray"):
                col_i = col_i.toarray().ravel()
                col_j = col_j.toarray().ravel()
            else:
                col_i = np.asarray(col_i).ravel()
                col_j = np.asarray(col_j).ravel()

            intersection = np.logical_and(col_i, col_j).sum()
            union = np.logical_or(col_i, col_j).sum()
            if union == 0:
                score = 0.0
            else:
                score = intersection / union
            jaccard_scores.append(score)
    mean_jaccard = np.mean(jaccard_scores) if jaccard_scores else 0.0
    return mean_jaccard


# def redundancy2(X):
#     """
#     X is already the selected_X
#     For each occurrence, counts how many features (columns) cover it (i.e., X[i, j] == 1).
#     Then, computes the mean redundancy over all positively covered occurrences.
#     Returns:
#         redundancy: average number of features covering each positive occurrence
#     """
#     # Count how many features cover each occurrence
#     coverage_counts = np.sum(X == 1, axis=1)
#     # Only consider occurrences that are covered by at least one feature
#     unique_occurrences = coverage_counts > 0
#     if np.any(unique_occurrences):
#         redundancy = np.mean(coverage_counts[unique_occurrences])
#     else:
#         redundancy = 0.0
#     return redundancy