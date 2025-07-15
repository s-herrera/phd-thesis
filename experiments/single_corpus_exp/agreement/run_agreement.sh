#!/bin/bash

# --------------------------------------------------------
# ES UPOS
# python3 ../grex2/extract_rules_via_lasso.py \
# ../data/sud-treebanks-v2.15/SUD_Spanish-AnCora/preprocessed_data/es_ancora-sud-train.conllu \
# --config sud \
# --patterns patterns/patterns_agreement_number_upos.txt \
# --output "res_es_number_upos_agreement.json" \
# --max-degree 3

# python3 ../grex2/extract_rules_via_lasso.py \
# ../data/sud-treebanks-v2.15/SUD_Spanish-AnCora/preprocessed_data/es_ancora-sud-train.conllu \
# --config sud \
# --patterns patterns/patterns_agreement_gender_upos.txt \
# --output "res_es_gender_upos_agreement.json" \
# --max-degree 3

# python3 ../grex2/extract_rules_via_lasso.py \
# ../data/sud-treebanks-v2.15/SUD_Spanish-AnCora/preprocessed_data/es_ancora-sud-train.conllu \
# --config sud \
# --patterns patterns/patterns_agreement_person_upos.txt \
# --output "res_es_person_upos_agreement.json" \
# --max-degree 3

# ---------------------
# ES with features

# python3 ../grex2/extract_rules_via_lasso.py \
#     ../data/sud-treebanks-v2.15/SUD_Spanish-AnCora/preprocessed_data/es_ancora-sud-train.conllu \
#     --config sud \
#     --patterns patterns/patterns_agreement_number.txt \
#     --output "agreement/results/res_es_number_agreement.json" \
#     --max-degree 2 

# python3 ../grex2/extract_rules_via_lasso.py \
#     ../data/sud-treebanks-v2.15/SUD_Spanish-AnCora/preprocessed_data/es_ancora-sud-train.conllu \
#     --config sud \
#     --patterns patterns/patterns_agreement_gender.txt \
#     --output "agreement/results/res_es_gender_agreement.json" \
#     --max-degree 2

# python3 ../grex2/extract_rules_via_lasso.py \
#     ../data/sud-treebanks-v2.15/SUD_Spanish-AnCora/preprocessed_data/es_ancora-sud-train.conllu \
#     --config sud \
#     --patterns patterns/patterns_agreement_person.txt \
#     --output "agreement/results/res_es_person_agreement.json" \
#     --max-degree 2