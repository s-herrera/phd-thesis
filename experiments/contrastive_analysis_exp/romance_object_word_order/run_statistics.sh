#!/bin/bash

# Statistics

# python3 compute_statistics.py \
# -i results/results_xy_order.pickle \
# --scope "X[upos]; X->Y" \
# --conclusion "X << Y" \
# --corpora bUD_Catalan-AnCora@2.15,UD_French-GSD@2.15,bUD_Italian-ISDT@2.15,UD_Portuguese-Porttinari@2.15,UD_Romanian-RRT@2.15,bUD_Spanish-AnCora@2.15

# # python3 compute_statistics.py \
# -i results/results_xy_obj_verb_order.pickle \
# --scope "X-[obj]->Y; X[upos=VERB]" \
# --conclusion "X << Y" \
# --corpora bUD_Catalan-AnCora@2.15,UD_French-GSD@2.15,bUD_Italian-ISDT@2.15,UD_Portuguese-Porttinari@2.15,UD_Romanian-RRT@2.15,bUD_Spanish-AnCora@2.15

# # python3 compute_statistics.py \
# -i results/results_xy_obj_verb_noun_order.pickle \
# --scope "X-[obj]->Y; X[upos=VERB]; Y[upos=NOUN]" \
# --conclusion "X << Y" \
# --corpora bUD_Catalan-AnCora@2.15,UD_French-GSD@2.15,bUD_Italian-ISDT@2.15,UD_Portuguese-Porttinari@2.15,UD_Romanian-RRT@2.15,bUD_Spanish-AnCora@2.15

# # python3 compute_statistics.py \
# -i results/results_xy_obj_verb_pron_order.pickle \
# --scope "X-[obj]->Y; X[upos=VERB]; Y[upos=PRON]" \
# --conclusion "X << Y" \
# --corpora bUD_Catalan-AnCora@2.15,UD_French-GSD@2.15,bUD_Italian-ISDT@2.15,UD_Portuguese-Porttinari@2.15,UD_Romanian-RRT@2.15,bUD_Spanish-AnCora@2.15

# No Catalan

python3 compute_statistics.py \
-i results/results_xy_order.pickle \
--scope "X[upos]; X->Y" \
--conclusion "X << Y" \
--corpora UD_French-GSD@2.15,bUD_Italian-ISDT@2.15,bUD_Portuguese-Porttinari@2.15,UD_Romanian-RRT@2.15,bUD_Spanish-AnCora@2.15,bUD_Catalan-AnCora@2.15

python3 compute_statistics.py \
-i results/results_xy_obj_verb_order_no_catalan.pickle \
--scope "X-[obj]->Y; X[upos=VERB]" \
--conclusion "X << Y" \
--corpora UD_French-GSD@2.15,bUD_Italian-ISDT@2.15,bUD_Portuguese-Porttinari@2.15,UD_Romanian-RRT@2.15,bUD_Spanish-AnCora@2.15

python3 compute_statistics.py \
-i results/results_xy_obj_verb_noun_order_no_catalan.pickle \
--scope "X-[obj]->Y; X[upos=VERB]; Y[upos=NOUN]" \
--conclusion "X << Y" \
--corpora UD_French-GSD@2.15,bUD_Italian-ISDT@2.15,bUD_Portuguese-Porttinari@2.15,UD_Romanian-RRT@2.15,bUD_Spanish-AnCora@2.15

python3 compute_statistics.py \
-i results/results_xy_obj_verb_pron_order_no_catalan.pickle \
--scope "X-[obj]->Y; X[upos=VERB]; Y[upos=PRON]" \
--conclusion "X << Y" \
--corpora UD_French-GSD@2.15,bUD_Italian-ISDT@2.15,bUD_Portuguese-Porttinari@2.15,UD_Romanian-RRT@2.15,bUD_Spanish-AnCora@2.15


# alpha 0.0001
# python3 compute_statistics.py \
# -i results/results_xy_obj_verb_order_alpha0001.pickle \
# --scope "X-[obj]->Y; X[upos=VERB]" \
# --conclusion "X << Y" \
# --corpora bUD_Catalan-AnCora@2.15,UD_French-GSD@2.15,bUD_Italian-ISDT@2.15,bUD_Portuguese-Porttinari@2.15,UD_Romanian-RRT@2.15,bUD_Spanish-AnCora@2.15
