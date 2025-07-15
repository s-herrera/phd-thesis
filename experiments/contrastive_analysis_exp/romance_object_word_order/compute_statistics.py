from grewpy import Request
from scipy.stats import chisquare
import numpy as np

import pickle
import argparse
import json
import re

def grewmatch_link(scope, pattern, conclusion, corpora):
    """
    Create a query on grew-match
    """
    base_url = "https://universal.grew.fr/?"
    formatted_pattern = "%0A".join([line.strip() for line in pattern.split("\n") if line.strip().startswith(("pattern", "with"))])
    url = f"{base_url}pattern={scope}%0A{formatted_pattern}&whether={conclusion}&corpus_list={corpora}"
    return url

def pattern_to_request(pattern, scope):
    """
    Build a Grew request from a Grex pattern
    """
    def parents_in_scope(scope: str) -> dict:
        """
        Get scope dependencies. Parent relations are needed to build a grew request.
        """
        parents = dict()
        for clause in Request(scope).json_data():
            for constraint in clause['pattern']: # type: ignore
                if "->" in constraint:
                    parent, child = re.split("-.*-?>", constraint)
                    parents[child] = parent
        return parents

    scope_parents = parents_in_scope(scope)
    request = Request()

    for clause in re.split(",node", pattern):
        _, node_name, target, feature = clause.split(":", maxsplit=3)
        keyword = "with" if target == "child" else "pattern"
        feat, value = feature.split("=")
        parent = scope_parents.get(node_name, f"{node_name}parent")
        # position
        if feat == "position":
            if value == "after":
                request.append(keyword, f"{parent}->{node_name}; {node_name} << {parent}")
            else:
                request.append(keyword, f"{parent}->{node_name}; {parent} << {node_name}")
        # deprel
        elif "rel_shallow" in feat:
            if target == "own":
                request.append(keyword, f'{parent}-[{value}]->{node_name}')
            else: #child
                request.append(keyword, f'{node_name}-[{value}]->{node_name}child')
        elif target == "child":
            request.append(keyword, f'{node_name}->{node_name}child; {node_name}child[{feat}="{value}"]')
        elif target == "parent":
            request.append(keyword, f'{parent}->{node_name}; {parent}[{feat}="{value}"]')
        else: #own
            request.append(keyword, f'{node_name}[{feat}="{value}"]')
    lst_request = [str(req) for req in request]
    return "\n".join(lst_request)

def variance(values):
    """
    Compute variance
    """
    res = np.array(values)
    return np.var(res, ddof=1)

def std(values):
    """
    Compute standard deviation
    """
    res = np.array(values)
    return np.std(res, ddof=1)

def cv(values):
    """
    Compute coefficient of variation.
    """
    # really sensentive when the mean is close to 0
    std = np.std(values, ddof=1)
    return std / np.mean(values) if np.mean(values) > 0 else 0

def cramers_v(chi2, n, df):
    """
    Compute Cramer's V effect size.
    """
    v = np.sqrt(chi2 / (n * df))
    return v

def convert_to_json_compatible(data):
    """
    Make data json compatible.
    """
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.generic):
        return data.item()
    elif isinstance(data, dict):
        return {key: convert_to_json_compatible(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_json_compatible(item) for item in data]
    else:
        return data

def compute_stats(file: pickle.Pickler, scope: str, conclusion: str, corpora: str = None) -> dict:
    """
    Compute stats given a set of contrastive patterns. Return a dictionary json compatible.
    file: pickle file from Grex2
    scope: scope of Grex2 extraction
    conclusion: conclusion of Grex2 extraction
    corpora: a set of corpus to create a grew-match link
    """

    with open(file, "rb") as f:
        extracted_rules = pickle.load(f)

    # values are computed over the vectors of the binary matrix used by the classifier.
    # target variable vector
    y = np.array(extracted_rules['y'], dtype=int)
    # mapping occurrences - languages
    language_vector = np.array(extracted_rules['languages'])
    # binary vector for each rule
    rule_vectors = [np.array(rule['matches'], dtype=int) for rule in extracted_rules['rules']]

    scope = f"pattern {{ {scope} }}"
    rule_names = [pattern_to_request(rule['pattern'], scope) for rule in extracted_rules['rules']]
    rule_alphas = [rule['alpha'] for rule in extracted_rules['rules']]

    if corpora:
        rule_urls = [grewmatch_link(scope, pattern, conclusion, corpora) for pattern in rule_names]
    else:
        rule_urls = [None] * len(rule_names)
    
    rule_occurrences = [rule['n_pattern_occurence'] for rule in extracted_rules['rules']]
    positive_occurrences = [rule['n_pattern_positive_occurence'] for rule in extracted_rules['rules']]
    negative_occurrences = (np.array(rule_occurrences) - np.array(positive_occurrences)).tolist()
    decisions = [rule['decision']  for rule in extracted_rules['rules']]
    unique_languages = np.unique(language_vector)

    rule_precision = []

    stats = {key: [] for key in [
        "p_lang_occs", "q_lang_occs", "p_q_lang_occs", "precisions", "cvs", 
        "contains_zero", "significances", "pvalues", "cramers_v", "residuals"]
        }

    for i in range(len(rule_vectors)):

        pattern_expected_occs = []
        pattern_observed_occs = []
        pattern_precisions = []
        rule_coverages = []
        rule_p_lang_occs = []
        rule_q_lang_occs = []
        rule_p_q_lang_occs = []
        expected_precision = []
        
        p_occs = np.sum(rule_vectors[i])
        not_p_occs = np.sum(np.logical_not(rule_vectors[i]))
        p_q_occs = np.sum(rule_vectors[i][y == 1])
        p_not_q_occs = np.sum(rule_vectors[i][y == 0])

        decision = decisions[i]
        if decision == "yes":
            rule_prec = p_q_occs / p_occs
            rule_precision.append(rule_prec)
        else:
            rule_prec = p_not_q_occs / p_occs
            rule_precision.append(rule_prec)

        for lang in unique_languages:
            lang_mask = language_vector == lang
            p_in_lang_occs = np.sum(rule_vectors[i][lang_mask])
            q_in_lang_occs = np.sum(y[lang_mask])
            not_p_in_lang_occs = np.sum(np.logical_not(rule_vectors[i])[lang_mask])
            not_q_in_lang_occs = np.sum(np.logical_not(y)[lang_mask])

            if decision == "yes":
                # observed (p_q_in_lang_occs) and expected
                observed_occs = np.sum(rule_vectors[i][lang_mask & (y == 1)])
                expected_occs = p_q_occs * (p_in_lang_occs / p_occs) # E = #p_and_q * (#p_for_lang / #p)
                prec = observed_occs / p_in_lang_occs if p_in_lang_occs > 0 else 0
                cov = observed_occs / q_in_lang_occs if q_in_lang_occs > 0  else 0
                rule_p_q_lang_occs.append(observed_occs)
                expected_precision.append(expected_occs / p_in_lang_occs if p_in_lang_occs > 0 else 0)
            else:
                #observed = p and not q
                observed_occs = np.sum(rule_vectors[i][lang_mask & (y == 0)])
                expected_occs = p_not_q_occs * (p_in_lang_occs / p_occs) # E = #p_and_q * (#p_for_lang / #p)
                prec = observed_occs / p_in_lang_occs if p_in_lang_occs > 0 else 0
                cov = observed_occs / not_q_in_lang_occs if q_in_lang_occs > 0  else 0
                rule_p_q_lang_occs.append(observed_occs)
                expected_precision.append(expected_occs / p_in_lang_occs if p_in_lang_occs > 0 else 0)

            pattern_observed_occs.append(observed_occs)
            pattern_expected_occs.append(expected_occs)
    
            pattern_precisions.append(prec)
            rule_coverages.append(cov)
            rule_p_lang_occs.append(p_in_lang_occs)
            rule_q_lang_occs.append(q_in_lang_occs)
              
        # compute chisquare, cramer's V, pvalues and residuals
        if pattern_observed_occs.count(0) == len(unique_languages) - 1:
            stats['significances'].append("unique")
            for key in ['pvalues', 'residuals', 'cramers_v']:
                stats[key].append(None)

        # check if >= 5 occurrences per cell and if observed counts >= 10
        elif all(occ >= 5 for occ in pattern_expected_occs if occ > 0) and np.sum(pattern_observed_occs) >= 10:
            # Filter out zero expected occurrences for score calculations
            non_zero_indices = [idx for idx, exp in enumerate(pattern_expected_occs) if exp > 0]
            non_zero_observed = [pattern_observed_occs[idx] for idx in non_zero_indices]
            non_zero_expected = [pattern_expected_occs[idx] for idx in non_zero_indices]
            chi_res = chisquare(non_zero_observed, non_zero_expected, sum_check=True)
            cramers_value = cramers_v(chi_res.statistic, sum(non_zero_observed), len(non_zero_observed) - 1)
            stats['significances'].append("distinctive" if chi_res.pvalue < 0.01/len(rule_names) else "common")
            stats['pvalues'].append(chi_res.pvalue)
            stats['cramers_v'].append(cramers_value)
            stats['residuals'].append([
                (obs - exp) / np.sqrt(exp) if exp > 0 else 0
                for obs, exp in zip(pattern_observed_occs, non_zero_expected)
            ])
        else:
            stats['significances'].append("low frequency")
            for key in ['pvalues', 'residuals', 'cramers_v']:
                stats[key].append(None)

        stats['contains_zero'].append(any([n == 0 for n in pattern_expected_occs]))
        stats['p_lang_occs'].append(rule_p_lang_occs)
        stats['q_lang_occs'].append(rule_q_lang_occs)
        stats['p_q_lang_occs'].append(rule_p_q_lang_occs)
        stats['precisions'].append(pattern_precisions)
        stats['cvs'].append(cv(pattern_precisions))

    json_data = {
        "scope": scope,
        "scope_occs": extracted_rules['data_len'],
        "conclusion": conclusion,
        "conclusion_occs": extracted_rules['n_yes'],
        "languages" : unique_languages,
        "rules": [
            {
                "rule": rule_names[i],
                "grew-match": rule_urls[i],
                "p occs": rule_occurrences[i],
                "pos occs": positive_occurrences[i],
                "neg occs": negative_occurrences[i],
                "rule_precision": rule_precision[i],
                "decision": "right" if decisions[i] == "yes" else "left",
                "alpha": rule_alphas[i],
                "stats": {
                    "p_lang_occs": stats["p_lang_occs"][i],
                    "q_lang_occs": stats["q_lang_occs"][i],
                    "p_q_lang_occs": stats["p_q_lang_occs"][i],
                    "precisions": stats["precisions"][i],
                    "cv": stats["cvs"][i],
                    "contains_zero": stats["contains_zero"][i],
                    "significance": stats["significances"][i],
                    "pvalue": stats["pvalues"][i],
                    "cramers_v": stats['cramers_v'][i],
                    "residuals": stats["residuals"][i],

                },
            }
            for i in range(len(rule_names))
        ]
    }

    compatible_json_data = convert_to_json_compatible(json_data)
    return compatible_json_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-i", "--input_file", required=True, help="Path to the input file")
    parser.add_argument("-s", "--scope", required=True, help="scope")
    parser.add_argument("-c", "--conclusion", required=True, help="conclusion")
    parser.add_argument("--corpora", required=False, help="corpora")

    args = parser.parse_args()
    filename = args.input_file.split(".")[0]
    res = compute_stats(args.input_file, args.scope, args.conclusion, args.corpora)
    with open(f"{filename}_statistics.json", "w") as json_file:
        json.dump(res, json_file, indent=4)
