# Predifined scopes

# python3 ../../grex2/extract_rules_via_lasso.py \
#     ../../data/SUD_French-Sequoia-r2.15/preprocessed_data \
#     --config sud \
#     --patterns patterns/patterns_adj_noun.txt \
#     --output "results/res_sequoia_wo_adj_noun.json" \
#     --max-degree 2 --alpha-start 0.001 --alpha-num 1

# python3 ../../grex2/extract_rules_via_lasso.py \
#     ../../data/SUD_French-Sequoia-r2.15/preprocessed_data \
#     --config sud \
#     --patterns patterns/patterns_adp_noun.txt \
#     --output "results/res_sequoia_wo_adp_noun.json" \
#     --max-degree 2 --alpha-start 0.001 --alpha-num 1

# python3 ../../grex2/extract_rules_via_lasso.py \
#     ../../data/SUD_French-Sequoia-r2.15/preprocessed_data \
#     --config sud \
#     --patterns patterns/patterns_advcl_sconj.txt \
#     --output "results/res_sequoia_wo_advcl_sconj.json" \
#     --max-degree 2 --alpha-start 0.001 --alpha-num 1

# python3 ../../grex2/extract_rules_via_lasso.py \
#     ../../data/SUD_French-Sequoia-r2.15/preprocessed_data \
#     --config sud \
#     --patterns patterns/patterns_dem_noun.txt \
#     --output "results/res_sequoia_wo_dem_noun.json" \
#     --max-degree 2 --alpha-start 0.001 --alpha-num 1

# python3 ../../grex2/extract_rules_via_lasso.py \
#     ../../data/SUD_French-Sequoia-r2.15/preprocessed_data \
#     --config sud \
#     --patterns patterns/patterns_num_noun.txt \
#     --output "results/res_sequoia_wo_num_noun.json" \
#     --max-degree 2 --alpha-start 0.001 --alpha-num 1

# python3 ../../grex2/extract_rules_via_lasso.py \
#     ../../data/SUD_French-Sequoia-r2.15/preprocessed_data \
#     --config sud \
#     --patterns patterns/patterns_obj_verb.txt \
#     --output "results/res_sequoia_wo_obj_verb.json" \
#     --max-degree 2 --alpha-start 0.001 --alpha-num 1

# python3 ../../grex2/extract_rules_via_lasso.py \
#     ../../data/SUD_French-Sequoia-r2.15/preprocessed_data \
#     --config sud \
#     --patterns patterns/patterns_relative_noun.txt \
#     --output "results/res_sequoia_wo_relative_noun.json" \
#     --max-degree 2 --alpha-start 0.001 --alpha-num 1

# python3 ../../grex2/extract_rules_via_lasso.py \
#     ../../data/SUD_French-Sequoia-r2.15/preprocessed_data \
#     --config sud \
#     --patterns patterns/patterns_subj_verb.txt \
#     --output "results/res_sequoia_wo_subj_verb.json" \
#     --max-degree 2 --alpha-start 0.001 --alpha-num 1

# low lambda

# python3 ../../grex2/extract_rules_via_lasso.py \
#     ../../data/SUD_French-Sequoia-r2.15/preprocessed_data \
#     --config sud \
#     --patterns patterns/patterns_word_order_low_lambda.txt \
#     --output "results/res_sequoia_wo_low_lamda.json" \
#     --max-degree 2 --alpha-start 0.0001 --alpha-num 1