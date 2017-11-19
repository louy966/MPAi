#!/bin/bash


stage=0
train_stage=-100
# This trains only unadapted (just cepstral mean normalized) features,
# and uses various combinations of VTLN warping factor and time-warping
# factor to artificially expand the amount of data.

. cmd.sh

. utils/parse_options.sh  # to parse the --stage option, if given

[ $# != 0 ] && echo "Usage: local/run_4b.sh [--stage <stage> --train-stage <train-stage>]" && exit 1;

set -e

if [ $stage -le 0 ]; then 
  # Create the training data.
  featdir=`pwd`/mfcc/nnet5b; mkdir -p $featdir
  fbank_conf=conf/fbank_40.conf

  steps/nnet2/get_perturbed_feats.sh --cmd "$train_cmd" \
    $fbank_conf $featdir exp/perturbed_fbanks data/train data/train_perturbed_fbank &
  steps/nnet2/get_perturbed_feats.sh --cmd "$train_cmd" --feature-type mfcc \
    conf/mfcc.conf $featdir exp/perturbed_mfcc data/train data/train_perturbed_mfcc &
  wait
fi

if [ $stage -le 1 ]; then
  steps/align_fmllr.sh --nj 4 --cmd "$train_cmd" \
    data/train_perturbed_mfcc data/lang_2 exp/tri4b exp/tri4b_ali_perturbed_mfcc
fi 

if [ $stage -le 2 ]; then
  steps/nnet2/train_block.sh --stage "$train_stage" \
     --cleanup false \
     --initial-learning-rate 0.01 --final-learning-rate 0.001 \
     --num-epochs 10 --num-epochs-extra 5 \
     --cmd "$decode_cmd" \
     --hidden-layer-dim 1536 \
     --num-block-layers 3 --num-normal-layers 3 \
      data/train_perturbed_fbank data/lang_2 exp/tri4b_ali_perturbed_mfcc exp/nnet5b  || exit 1
fi

if [ $stage -le 3 ]; then # create testing fbank data.
  featdir=`pwd`/mfcc
  fbank_conf=conf/fbank_40.conf

    rm -r data/test_fbank
    cp -r data/test data/test_fbank
    rm -r test_fbank/split* || true
    steps/make_fbank.sh --fbank-config "$fbank_conf" --nj 4 \
      --cmd "$train_cmd" data/test_fbank exp/make_fbank/test $featdir  || exit 1;
    steps/compute_cmvn_stats.sh data/test_fbank exp/make_fbank/test $featdir  || exit 1;
  done
fi

if [ $stage -le 4 ]; then
  steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 4 \
     exp/tri4b/graph data/test_fbank exp/nnet5b_gpu/decode

fi



exit 0;

