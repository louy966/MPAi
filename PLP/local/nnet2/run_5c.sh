#!/bin/bash

# This is neural net training on top of adapted 40-dimensional features.
#

train_stage=-10
use_gpu=true

. cmd.sh
. ./path.sh
. utils/parse_options.sh


if $use_gpu; then
  if ! cuda-compiled; then
    cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
  fi
  parallel_opts="--gpu 1"
  num_threads=1
  minibatch_size=512
  dir=exp/nnet5c_gpu
else
  num_threads=16
  parallel_opts="--num-threads $num_threads"
  dir=exp/nnet5c
  minibatch_size=128
fi

if [ ! -f $dir/final.mdl ]; then
  steps/nnet2/train_tanh_fast.sh --stage $train_stage \
    --num-threads "$num_threads" \
    --parallel-opts "$parallel_opts" \
    --minibatch-size "$minibatch_size" \
    --num-jobs-nnet 8 \
    --samples-per-iter 400000 \
    --mix-up 8000 \
    --initial-learning-rate 0.01 --final-learning-rate 0.001 \
    --num-hidden-layers 4 --hidden-layer-dim 1024 \
    --cmd "$decode_cmd" \
     data/train data/lang_2 exp/tri4b $dir || exit 1
fi


steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 4 \
  --transform-dir exp/tri4b/decode \
   exp/tri4b/graph data/test $dir/decode

wait
