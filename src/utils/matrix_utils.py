import yaml
import sys
import numpy as np

import pyximport
pyximport.install()

from grex2.grex.data import extract_data
from grex2.grex.utils import FeaturePredicate
import grex2.grex.features
from scipy.sparse import hstack, csr_matrix
import re

def build_occurrences(corpus, patterns, grew_config="ud"):

    with open(patterns) as instream:
        config = yaml.load(instream, Loader=yaml.Loader)

    scope = config["scope"]
    conclusion = config.get("conclusion", None)
    conclusion_meta = config.get("conclusion_meta", None)

    templates = FeaturePredicate.from_config(config["templates"])
    feature_predicate = FeaturePredicate.from_config(config["features"], templates=templates)

    print("Loading dataset...", flush=True)
    data = extract_data(corpus, scope, conclusion, conclusion_meta, feature_predicate, grew_config)
    return data


def matrix(data, max_degree = 2, min_feature_occurence = 5):

    data_inputs = list()
    data_outputs = list()
    for sentence in data:
        data_inputs.append(sentence["input"])
        data_outputs.append(sentence["output"])

    n_positive = sum(data_outputs)
    print("Number of occurences of the conclusion: %i / %i" % (n_positive, len(data)))
    print("Extracting features", flush=True)
    feature_set = grex2.grex.features.FeatureSet()

    feature_set.add_feature(grex2.grex.features.AllSingletonFeatures())
    for degree in range(2, max_degree + 1):
        feature_set.add_feature(grex2.grex.features.AllProductFeatures(
            degree=degree,
            min_occurences=min_feature_occurence
        ))
    try:
        feature_set.init_from_data(data_inputs)
        feature_names = [name for fset in feature_set.features for name in fset.get_all_names()]
        X = feature_set.build_features(data_inputs, sparse=True)
        if X.shape[1] == 0:
            raise RuntimeError("Empty feature list!")
    except RuntimeError:
        RuntimeError("There was an error during feature extraction")

    # this is wrong, but it fix the bug with the min_occurrence filter
    col_counts = np.array((X != 0).sum(axis=0)).flatten()
    keep_cols = np.where(col_counts >= min_feature_occurence)[0]
    X = X[:, keep_cols]
    feature_names = [feature_names[i] for i in keep_cols] 

    y = np.empty((len(data),), dtype=np.int_)
    for i, v in enumerate(data_outputs):
        assert v in [0, 1]
        y[i] = v
    return X, y, feature_names

def align_matrix(X_selected, X_to_align, selected_feature_names, test_features):
    """
    Align X_b with X_a
    """
    selected_feature_sets = [set(re.split(r",(?=node:)", f)) for f in selected_feature_names]
    test_feature_sets = [set(re.split(r",(?=node:)", f)) for f in test_features]

    overlap = [f for f in selected_feature_sets if f in test_feature_sets]
    print(f"\nShared selected features in test set: {len(overlap)} / {len(selected_feature_sets)}\n")
    test_feature_map = {frozenset(feat): i for i, feat in enumerate(test_feature_sets)}
    # align X_b with X_a
    aligned_columns = []
    missing_features = []
    for feature in selected_feature_sets:
        key = frozenset(feature)
        if key in test_feature_map:
            idx = test_feature_map[key]
            aligned_columns.append(X_to_align[:, idx])
        else:
            aligned_columns.append(csr_matrix((X_to_align.shape[0], 1)))
            missing_features.append(",".join(sorted(feature)))

    X_aligned = hstack(aligned_columns, format="csr")
    assert X_aligned.shape[1] == X_selected.shape[1]
    return X_aligned, missing_features
