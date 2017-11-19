#!/bin/bash
. ./cmd.sh

steps/make_denlats.sh --nj 4 --sub-split 4 --cmd "$train_cmd" \
  --transform-dir exp/tri4b_ali \
  data/train data/lang_2 exp/tri4b exp/tri4b_denlats || exit 1;

steps/train_mmi.sh --cmd "$train_cmd" --boost 0.1 \
  data/train data/lang_2 exp/tri4b_ali exp/tri4b_denlats \
  exp/tri4b_mmi_b0.1  || exit 1;

steps/decode.sh --nj 4 --cmd "$decode_cmd" --transform-dir exp/tri4b/decode \
  exp/tri4b/graph data/test exp/tri4b_mmi_b0.1/decode

#first, train UBM for fMMI experiments.
steps/train_diag_ubm.sh --silence-weight 0.5 --nj 4 --cmd "$train_cmd" \
  600 data/train data/lang_2 exp/tri4b_ali exp/dubm4b

# Next, fMMI+MMI.
steps/train_mmi_fmmi.sh \
  --boost 0.1 --cmd "$train_cmd" data/train data/lang_2 exp/tri4b_ali exp/dubm4b exp/tri4b_denlats \
  exp/tri4b_fmmi_a || exit 1;

for iter in 3 4 5 6 7 8; do
 steps/decode_fmmi.sh --nj 4  --cmd "$decode_cmd" --iter $iter \
   --transform-dir exp/tri4b/decode  exp/tri4b/graph data/test \
  exp/tri4b_fmmi_a/decode_it$iter &
done

# fMMI + mmi with indirect differential.
steps/train_mmi_fmmi_indirect.sh \
  --boost 0.1 --cmd "$train_cmd" data/train data/lang_2 exp/tri4b_ali exp/dubm4b exp/tri4b_denlats \
  exp/tri4b_fmmi_indirect || exit 1;

for iter in 3 4 5 6 7 8; do
 steps/decode_fmmi.sh --nj 4  --cmd "$decode_cmd" --iter $iter \
   --transform-dir exp/tri4b/decode  exp/tri4b/graph data/test \
  exp/tri4b_fmmi_indirect/decode_it$iter &
done

