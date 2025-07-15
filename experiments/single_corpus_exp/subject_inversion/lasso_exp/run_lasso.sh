# -----------------------------
# Check search space 
# -----------------------------

# python3 -m grex2.check_features \
# ../../../../data/single_corpus_data/SUD_French-GSD-r2.15/preprocessed_data \
# --patterns patterns/patterns_subject_inversion.txt \
# --config sud \

# -----------------------------
# Rule extraction 
# -----------------------------

# python3 -m grex2.extract_rules_via_lasso \
# ../../../../data/single_corpus_data/SUD_French-GSD-r2.15/preprocessed_data \
# --config sud \
# --patterns ../../patterns_single_corpus/patterns_subject_inversion.txt \
# --output ../../../../results/results_subject_inversion_lasso_degreea1.json \
# --max-degree 1 \

python3 -m grex2.extract_rules_via_lasso \
../../../../data/single_corpus_data/SUD_French-GSD-r2.15/preprocessed_data \
--config sud \
--patterns ../../patterns_single_corpus/patterns_subject_inversion.txt \
--output ../../../../results/results_subject_inversion_lasso_degree2.json \
--max-degree 2 \

# python3 -m grex2.extract_rules_via_rulefit \
# ../../../../data/single_corpus_data/SUD_French-GSD-r2.15/preprocessed_data \
# --config sud \
# --patterns ../../patterns_single_corpus/patterns_subject_inversion.txt \
# --output ../../../../results/results_subject_inversion_rulefit.json \

# -----------------------------
# Global measures and evaluation
# -----------------------------

# python -m evaluation.l1_model_evaluation \
# --path ../../../../data/single_corpus_data/SUD_French-GSD-r2.15/preprocessed_data \
# --patterns ../../patterns_single_corpus/patterns_subject_inversion.txt \
# --config sud \
# --max-degree 1 \
# --output evaluation_subject_inversion_lasso_degree1.json

# python -m evaluation.l1_model_evaluation \
# --path ../../../../data/single_corpus_data/SUD_French-GSD-r2.15/preprocessed_data \
# --patterns ../../patterns_single_corpus/patterns_subject_inversion.txt \
# --config sud \
# --max-degree 2 \
# --output evaluation_subject_inversion_lasso_degree1.json

# -----------------------------
# V2.5
# -----------------------------
# !!! To compare results with Chaudhary et al.'s work it's necessary to not balance the classes in the evaluation (class_weight='balanced')
# python -m evaluation.l1_model_evaluation \
# --path ../../../../data/single_corpus_data/SUD_French-GSD-r2.5 \
# --patterns ../../patterns_single_corpus/patterns_subject_inversion_v2.5.txt \
# --config sud \
# --max-degree 2 \
# --output evaluation_v2.5_subject_inversion_lasso_degree2.json