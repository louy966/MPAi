#!/bin/bash

. ./path.sh || exit 1
. ./cmd.sh || exit 1

nj=4       # number of parallel jobs - 1 is perfect for such a small data set
lm_order=1 # language model order (n-gram quantity) - 1 is enough for digits grammar

echo
echo "===== nnet2 ====="
echo

local/run_nnet2.sh

echo
echo "===== run3.sh script is finished ====="
echo
