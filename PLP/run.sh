#!/bin/bash

. ./path.sh || exit 1
. ./cmd.sh || exit 1

nj=4       # number of parallel jobs - 1 is perfect for such a small data set
lm_order=1 # language model order (n-gram quantity) - 1 is enough for digits grammar

# Safety mechanism (possible running this script with modified arguments)
. utils/parse_options.sh || exit 1
[[ $# -ge 1 ]] && { echo "Wrong arguments!"; exit 1; }

# Removing previously created data (from last run.sh execution)
rm -rf exp plp data/train/spk2utt data/train/cmvn.scp data/train/feats.scp data/train/split1 data/test/spk2utt data/test/cmvn.scp data/test/feats.scp data/test/split1 data/local/lang data/lang data/local/tmp data/local/dict/lexiconp.txt data/local/dict_2 data/local/lang_tmp data/lang_2 data/lang_test

echo
echo "===== PREPARING ACOUSTIC DATA ====="
echo

# Needs to be prepared by hand (or using self written scripts):
#
# spk2gender  [<speaker-id> <gender>]
# wav.scp     [<uterranceID> <full_path_to_audio_file>]
# text           [<uterranceID> <text_transcription>]
# utt2spk     [<uterranceID> <speakerID>]
# corpus.txt  [<text_transcription>]

# Making spk2utt files
utils/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt
utils/utt2spk_to_spk2utt.pl data/test/utt2spk > data/test/spk2utt

echo
echo "===== FEATURES EXTRACTION ====="
echo

# Making feats.scp files
plpdir=plp
# Uncomment and modify arguments in scripts below if you have any problems with data sorting
# utils/validate_data_dir.sh data/train     # script for checking prepared data - here: for data/train directory
# utils/fix_data_dir.sh data/train          # tool for data proper sorting if needed - here: for data/train directory
steps/make_plp.sh --nj $nj --cmd "$train_cmd" data/train exp/make_plp/train $plpdir
steps/make_plp.sh --nj $nj --cmd "$train_cmd" data/test exp/make_plp/test $plpdir

# Making cmvn.scp files
steps/compute_cmvn_stats.sh data/train exp/make_plp/train $plpdir
steps/compute_cmvn_stats.sh data/test exp/make_plp/test $plpdir

echo
echo "===== PREPARING LANGUAGE DATA ====="
echo

# Needs to be prepared by hand (or using self written scripts):
#
# lexicon.txt           [<word> <phone 1> <phone 2> ...]
# nonsilence_phones.txt    [<phone>]
# silence_phones.txt    [<phone>]
# optional_silence.txt  [<phone>]

# Preparing language data
utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang data/lang

echo
echo "===== LANGUAGE MODEL CREATION ====="
echo "===== MAKING lm.arpa ====="
echo

loc=`which ngram-count`;
if [ -z $loc ]; then
   if uname -a | grep 64 >/dev/null; then
           sdir=$KALDI_ROOT/tools/srilm/bin/i686-m64
   else
                   sdir=$KALDI_ROOT/tools/srilm/bin/i686
   fi
   if [ -f $sdir/ngram-count ]; then
                   echo "Using SRILM language modelling tool from $sdir"
                   export PATH=$PATH:$sdir
   else
                   echo "SRILM toolkit is probably not installed.
                           Instructions: tools/install_srilm.sh"
                   exit 1
   fi
fi

local=data/local
mkdir $local/tmp
ngram-count -order $lm_order -write-vocab $local/tmp/vocab-full.txt -wbdiscount -text $local/corpus.txt -lm $local/tmp/lm.arpa

echo
echo "===== MAKING G.fst ====="
echo

lang=data/lang
arpa2fst --disambig-symbol=#0 --read-symbol-table=$lang/words.txt $local/tmp/lm.arpa $lang/G.fst

echo
echo "===== MONO TRAINING ====="
echo

steps/train_mono.sh --nj $nj --cmd "$train_cmd" data/train data/lang exp/mono  || exit 1

echo
echo "===== MONO DECODING ====="
echo

utils/mkgraph.sh --mono data/lang exp/mono exp/mono/graph || exit 1
steps/decode.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" exp/mono/graph data/test exp/mono/decode

echo
echo "===== MONO ALIGNMENT ====="
echo

steps/align_si.sh --nj $nj --cmd "$train_cmd" data/train data/lang exp/mono exp/mono_ali || exit 1

echo
echo "===== TRI1 (first triphone pass) TRAINING ====="
echo

steps/train_deltas.sh --cmd "$train_cmd" 2000 10000 data/train data/lang exp/mono_ali exp/tri1 || exit 1

echo
echo "===== TRI1 (first triphone pass) DECODING ====="
echo

utils/mkgraph.sh data/lang exp/tri1 exp/tri1/graph || exit 1
steps/decode.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" exp/tri1/graph data/test exp/tri1/decode

echo
echo "===== TRI1 ALIGNMENT ====="
echo

steps/align_si.sh --nj $nj --cmd "$train_cmd" \
  --use-graphs true data/train data/lang exp/tri1 exp/tri1_ali

echo
echo "===== tri2b [LDA+MLLT] TRAINING ====="
echo

steps/train_lda_mllt.sh --cmd "$train_cmd" \
  --splice-opts "--left-context=3 --right-context=3" \
 2500 15000 data/train data/lang exp/tri1_ali exp/tri2b

echo
echo "===== tri2b [LDA+MLLT] DECODING ====="
echo

utils/mkgraph.sh data/lang exp/tri2b exp/tri2b/graph
steps/decode.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" \
   exp/tri2b/graph data/test exp/tri2b/decode

echo
echo "===== tri2b ALIGNMENT ====="
echo

steps/align_si.sh --nj $nj --cmd "$train_cmd" --use-graphs true \
   data/train data/lang exp/tri2b exp/tri2b_ali

#echo
#echo "===== Do MMI on top of LDA+MLLT ====="
#echo

#steps/make_denlats.sh --nj $nj --cmd "$train_cmd" \
#  data/train data/lang exp/tri2b exp/tri2b_denlats
#steps/train_mmi.sh data/train data/lang exp/tri2b_ali exp/tri2b_denlats exp/tri2b_mmi
#steps/decode.sh --config conf/decode.config --iter 4 --nj $nj --cmd "$decode_cmd" \
#   exp/tri2b/graph data/test exp/tri2b_mmi/decode_it4
#steps/decode.sh --config conf/decode.config --iter 3 --nj $nj --cmd "$decode_cmd" \
#   exp/tri2b/graph data/test exp/tri2b_mmi/decode_it3

#echo
#echo "===== Do the same with boosting ====="
#echo

#steps/train_mmi.sh --boost 0.05 data/train data/lang \
#   exp/tri2b_ali exp/tri2b_denlats exp/tri2b_mmi_b0.05
#steps/decode.sh --config conf/decode.config --iter 4 --nj $nj --cmd "$decode_cmd" \
#   exp/tri2b/graph data/test exp/tri2b_mmi_b0.05/decode_it4
#steps/decode.sh --config conf/decode.config --iter 3 --nj $nj --cmd "$decode_cmd" \
#   exp/tri2b/graph data/test exp/tri2b_mmi_b0.05/decode_it3

#echo
#echo "===== Do MPE ====="
#echo

#steps/train_mpe.sh data/train data/lang exp/tri2b_ali exp/tri2b_denlats exp/tri2b_mpe
#steps/decode.sh --config conf/decode.config --iter 4 --nj $nj --cmd "$decode_cmd" \
#   exp/tri2b/graph data/test exp/tri2b_mpe/decode_it4
#steps/decode.sh --config conf/decode.config --iter 3 --nj $nj --cmd "$decode_cmd" \
#   exp/tri2b/graph data/test exp/tri2b_mpe/decode_it3

echo
echo "===== tri3b [LDA+MLLT+SAT] TRAINING ====="
echo

steps/train_sat.sh 4200 40000 data/train data/lang exp/tri2b_ali exp/tri3b

echo
echo "===== tri3b [LDA+MLLT+SAT] DECODING ====="
echo

utils/mkgraph.sh data/lang exp/tri3b exp/tri3b/graph
steps/decode_fmllr.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" \
  exp/tri3b/graph data/test exp/tri3b/decode

# (
#  utils/mkgraph.sh data/lang_ug exp/tri3b exp/tri3b/graph_ug
#  steps/decode_fmllr.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" \
#    exp/tri3b/graph_ug data/test exp/tri3b/decode_ug
# )

echo
echo "===== tri3b ALIGNMENT ====="
echo

steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" --use-graphs true \
  data/train data/lang exp/tri3b exp/tri3b_ali

echo
echo "===== MMI on top of tri3b (i.e. LDA+MLLT+SAT+MMI) ====="
echo

steps/make_denlats.sh --config conf/decode.config \
   --nj $nj --cmd "$train_cmd" --transform-dir exp/tri3b_ali \
  data/train data/lang exp/tri3b exp/tri3b_denlats
steps/train_mmi.sh data/train data/lang exp/tri3b_ali exp/tri3b_denlats exp/tri3b_mmi

steps/decode_fmllr.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" \
  --alignment-model exp/tri3b/final.alimdl --adapt-model exp/tri3b/final.mdl \
   exp/tri3b/graph data/test exp/tri3b_mmi/decode

# echo
# echo "===== Do a decoding that uses the exp/tri3b/decode directory to get transforms from ====="
# echo

# steps/decode.sh --config conf/decode.config --nj $nj --cmd "$decode_cmd" \
#   --transform-dir exp/tri3b/decode  exp/tri3b/graph data/test exp/tri3b_mmi/decode2

echo
echo "===== Estimate pronunciation and silence probabilities ====="
echo

  steps/get_prons.sh --cmd "$train_cmd" \
    data/train data/lang exp/tri3b || exit 1;
  utils/dict_dir_add_pronprobs.sh --max-normalize true \
    data/local/dict \
    exp/tri3b/pron_counts_nowb.txt exp/tri3b/sil_counts_nowb.txt \
    exp/tri3b/pron_bigram_counts_nowb.txt data/local/dict_2 || exit 1

  utils/prepare_lang.sh data/local/dict_2 \
    "<UNK>" data/local/lang_tmp data/lang_2 || exit 1;

    mkdir -p data/lang_test
    cp -r data/lang_2/* data/lang_test/ || exit 1;
    rm -rf data/lang_test/tmp
    cp data/lang/G.* data/lang_test/

echo
echo "===== train another SAT system (tri4b) ====="
echo

steps/train_sat.sh  --cmd "$train_cmd" 4200 40000 \
      data/train data/lang_2 exp/tri3b exp/tri4b || exit 1;

utils/mkgraph.sh data/lang_test \
      exp/tri4b exp/tri4b/graph || exit 1;

steps/decode_fmllr.sh --nj $nj --cmd "$decode_cmd" \
        exp/tri4b/graph data/test \
        exp/tri4b/decode || exit 1;
      steps/lmrescore.sh --cmd "$decode_cmd" \
        data/lang_test \
        data/test exp/tri4b/decode || exit 1

echo
echo "===== run.sh script is finished ====="
echo
