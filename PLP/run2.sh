#!/bin/bash

. ./path.sh || exit 1
. ./cmd.sh || exit 1

nj=4       # number of parallel jobs - 1 is perfect for such a small data set
lm_order=1 # language model order (n-gram quantity) - 1 is enough for digits grammar

echo
echo "===== Train and test MMI, and boosted MMI, on tri4b (LDA+MLLT+SAT) ====="
echo

steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
  data/train data/lang_2 exp/tri4b exp/tri4b_ali || exit 1;
local/run_mmi_tri4b.sh

echo
echo "===== run2.sh script is finished ====="
echo
