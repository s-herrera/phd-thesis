import json
import argparse
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn import tree
from utils.matrix_utils import build_occurrences, matrix

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Decision tree grid search for subject inversion experiment.")
    parser.add_argument('--input', type=str, required=True, help='Path to the input data file')
    parser.add_argument('--patterns', type=str, required=True, help='Path to save the results')
    parser.add_argument('--output', type=str, required=True, help='Path to the output data file')
    parser.add_argument('--grew-config', choices=["ud", "sud"], default="ud")
    parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility')

    args = parser.parse_args()

    data = build_occurrences(args.input, args.patterns, args.grew_config)
    X, y, feature_names = matrix(data, max_degree=1, min_feature_occurence=5)

    scores = ["balanced_accuracy", "f1_macro"]
    all_results = {s: dict() for s in scores}

    for score in scores:
        print(f"Starting gridsearch for {score} score")
        clf = tree.DecisionTreeClassifier(random_state=42)
        path = clf.cost_complexity_pruning_path(X, y)
        param_grid = {
            'criterion': ['entropy', 'gini'],
            'max_depth': [i for i in range(5, 15)],
            'ccp_alpha': np.linspace(0, path.ccp_alphas.max(), num=20)
        }

        grid_search = GridSearchCV(clf, param_grid, cv=5, scoring=score)
        grid_search.fit(X, y)

        # print("best estimator found:")
        # print(grid_search.best_estimator_)
        # print("\nbest parameters found:")
        # print(grid_search.best_params_)
        # print(f"\nbest precision score: {grid_search.best_score_:.4f}")

        results = {
            "best_estimator": str(grid_search.best_estimator_),
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "scoring": score
        }

        all_results[score] = results

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=4)