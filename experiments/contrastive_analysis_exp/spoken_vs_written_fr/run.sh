

python3 ../../grex2/extract_rules_via_lasso.py \
data/sequoia_parisstories_root_verbs_1490.conllu \
--config ud \
--patterns patterns/patterns_spoken_written.txt \
--output results/results_spoken_written_degree2.json \
--max-degree 2

# EVAL

# python3 ../../l1_model_evaluation.py \
# data/sequoia_parisstories_root_verbs_1490.conllu \
# --config ud \
# --patterns patterns/patterns_spoken_written.txt \
# --output results/eval_spoken_written_degree2.json \
# --max-degree 2