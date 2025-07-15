# -----------------------------
# Rule extraction 
# -----------------------------
# python3 -m grex2.extract_rules_via_dtree \
# ../../../../data/single_corpus_data/SUD_French-GSD-r2.15/preprocessed_data \
# --config sud \
# --patterns ../../patterns_single_corpus/patterns_subject_inversion.txt \
# --output ../../../../results/results_subject_inversion_dtree_depth8.json \
# --max-depth 8 \
# --only-leaves \
# --grew

# python3 -m grex2.extract_rules_via_dtree \
# ../../../../data/single_corpus_data/SUD_French-GSD-r2.15/preprocessed_data \
# --config sud \
# --patterns ../../patterns_single_corpus/patterns_subject_inversion.txt \
# --output ../../../../results/results_subject_inversion_dtree_depth10.json \
# --max-depth 10 \
# --only-leaves \
# --grew

# python3 -m grex2.extract_rules_via_dtree \
# ../../../../data/single_corpus_data/SUD_French-GSD-r2.15/preprocessed_data \
# --config sud \
# --patterns ../../patterns_single_corpus/patterns_subject_inversion.txt \
# --output ../../../../results/results_subject_inversion_dtree_depth14.json \
# --max-depth 14 \
# --only-leaves \
# --grew

# -----------------------------
# Comparison to Chaudhary et al.'s work 
# -----------------------------

# python3 -m grex2.extract_rules_via_dtree \
# ../../../../data/single_corpus_data/SUD_French-GSD-r2.5 \
# --config sud \
# --patterns ../../patterns_single_corpus/patterns_subject_inversion.txt \
# --output ../../../../results/results_v2.5_subject_inversion_dtree_depth8.json \
# --max-depth 8 \
# --only-leaves \
# --grew

# python3 -m grex2.extract_rules_via_dtree \
# ../../../../data/single_corpus_data/SUD_French-GSD-r2.5 \
# --config sud \
# --patterns ../../patterns_single_corpus/patterns_subject_inversion.txt \
# --output ../../../../results/results_v2.5_subject_inversion_dtree_depth14.json \
# --max-depth 14 \
# --only-leaves \
# --grew

# -----------------------------
# GridSearch 
# -----------------------------
# python3 dtree_gridsearch.py \
# --input ../../../../data/single_corpus_data/SUD_French-GSD-r2.15/preprocessed_data \
# --patterns ../../patterns_single_corpus/patterns_subject_inversion.txt \
# --output dtree_gridsearch_results.json