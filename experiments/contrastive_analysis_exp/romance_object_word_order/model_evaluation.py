
import sys
sys.path.insert(1, 'grex2')
import pyximport
pyximport.install()
import yaml
import grex.data
import grex.features
from grex.utils import FeaturePredicate
import numpy as np
import yaml
import skglm
import argparse
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score, accuracy_score
from scipy.sparse import hstack, csr_matrix

def extract_features(path, patterns):
    with open(patterns) as instream:
        config = yaml.load(instream, Loader=yaml.Loader)
        scope = config["scope"]
        conclusion = config.get("conclusion", None)
        conclusion_meta = config.get("conclusion_meta", None)
        templates = FeaturePredicate.from_config(config["templates"])
        feature_predicate = FeaturePredicate.from_config(config["features"], templates=templates)

    data, _ = grex.data.extract_data(path, scope, conclusion, conclusion_meta, feature_predicate)
    inputs = [sentence["input"] for sentence in data]
    outputs = [sentence["output"] for sentence in data]

    train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(
        inputs, outputs, test_size=0.25, stratify=outputs, random_state=42
    )
    return train_inputs, test_inputs, train_outputs, test_outputs

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    return acc, prec, recall, f1, balanced_accuracy

def build_X_y(input_data, target_data):
    feature_set = grex.features.FeatureSet()
    feature_set.add_feature(grex.features.AllSingletonFeatures())
    for degree in range(2, 2 + 1):
        feature_set.add_feature(grex.features.AllProductFeatures(
            degree=degree,
            min_occurences=10
        ))
    try:
        feature_set.init_from_data(input_data)
        feature_names = [name for fset in feature_set.features for name in fset.get_all_names()]
        # build features
        X = feature_set.build_features(input_data, sparse=True)
        if X.shape[1] == 0:
            raise RuntimeError("Empty feature list!")
    except RuntimeError:
        raise RuntimeError("There was an error during feature extraction")

    # build targets
    y = np.empty((len(input_data),), dtype=np.int_)
    for i, v in enumerate(target_data):
        assert v in [0, 1]
        y[i] = v
    return X, y, feature_names

def align_matrix(X_selected, X_to_align, feature_mask, features_a, features_b):
    """
    Align X_b with X_a
    """
    train_selected_feature_names = [features_a[i] for i in range(len(features_a)) if feature_mask[i]]
    overlap = [f for f in train_selected_feature_names if f in features_b]
    print(f"Shared selected features in test set: {len(overlap)} / {len(train_selected_feature_names)}\n")

    # align X_b with X_a
    aligned_columns = []
    for feature in train_selected_feature_names:
        if feature in features_b:
            # add feature column
            idx = features_b.index(feature)
            aligned_columns.append(X_to_align[:, idx])
        else:
            # add zero column
            aligned_columns.append(csr_matrix((X_to_align.shape[0], 1)))
    X_aligned = hstack(aligned_columns, format='csr')
    assert X_aligned.shape[1] == X_selected.shape[1] # check

    return X_aligned
    
def eval_train_test(path, patterns):

    train_inputs, test_inputs, train_outputs, test_outputs = extract_features(path, patterns)
    X, y, train_features = build_X_y(train_inputs, train_outputs)
    X_test, y_test, test_features = build_X_y(test_inputs, test_outputs)
    majority_class_baseline = max(np.sum(y)/len(y), np.sum(np.logical_not(y))/len(y))

    # model A to select features
    model_A = skglm.SparseLogisticRegression(
        alpha=0.001,
        fit_intercept=True,
        max_iter=20,
        max_epochs=1000,
    )

    # fit the model to select features
    model_A.fit(X, y)

    # we align both matrices to have the same features, the selected features by the first model.
    selector = SelectFromModel(model_A, prefit=True)
    X_selected = selector.transform(X)
    train_selected_features_mask = selector.get_support()
    X_test_aligned = align_matrix(X_selected, X_test, train_selected_features_mask, train_features, test_features)

    # Model B with the selected features
    model_B = skglm.SparseLogisticRegression(
        alpha=0.001,
        fit_intercept=True,
        max_iter=20,
        max_epochs=1000,
    )

    # train now only with selected features
    model_B.fit(X_selected, y)

    print("Results:")
    print(f"Majority-class baseline acc: {majority_class_baseline:.4f}\n")

    y_pred = model_B.predict(X_selected)
    train_metrics = compute_metrics(y, y_pred)
    print("Train (selected features):")
    print(f"Accuracy: {train_metrics[0]:.4f}")
    print(f"Precision: {train_metrics[1]:.4f}")
    print(f"Recall: {train_metrics[2]:.4f}")
    print(f"F1 Score: {train_metrics[3]:.4f}")
    print(f"Balanced Accuracy: {train_metrics[4]:.4f}")

    y_test_pred = model_B.predict(X_test_aligned)
    test_metrics = compute_metrics(y_test, y_test_pred)
    print("Test (selected features):")
    print(f"Accuracy: {test_metrics[0]:.4f}")
    print(f"Precision: {test_metrics[1]:.4f}")
    print(f"Recall: {test_metrics[2]:.4f}")
    print(f"F1 Score: {test_metrics[3]:.4f}")
    print(f"Balanced Accuracy: {test_metrics[4]:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--path", required=False, help="Path to the input file")
    parser.add_argument("--patterns", required=False, help="Grex pattern file")
    args = parser.parse_args()

    paths = [
        "data/xupos_xy_20900_42rs_nomisc.conllu",
        "data/x_obj_y_xverb_7250_nomisc_no_catalan.conllu",
        "data/x_obj_y_ynoun_xverb_5800_nomisc_no_catalan.conllu",
        "data/x_obj_y_ypron_xverb_910_nomisc_no_catalan.conllu"
        "../spoken_written_french/data/sequoia_parisstories_root_verbs_1490.conllu"
        ]
    
    patterns = [
        "patterns/patterns_xy_order.yml",
        "patterns/patterns_xy_obj_verb_order.yml",
        "patterns/patterns_xy_obj_verb_noun_order.yml",
        "patterns/patterns_xy_obj_verb_pron_order.yml"\
        "../spoken_written_french/patterns/patterns_spoken_written.txt"
        ]

    for path, pattern in zip(paths, patterns):
        eval_train_test(path, pattern)
