#!/bin/bash -l

set -eu
. ./path.sh
module list
export PYTHONIOENCODING='utf-8'

nbest_list=$1
am_score_file=$2
lm_score_file=$3
trn_dir=$4

if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <nbest-list> <acoustic-model-scores> <language-model-score-file> <trn-dir>" >&2
  exit 1
fi

mkdir -p ${trn_dir}

base=$(basename $lm_score_file); echo $base
for weight in $(seq 8 12); do
    result_file=${trn_dir}/hypoth_"$base"_lms"$weight".txt
    python3 ../../kaldi-utensils/cutlery/rescore_nbest.py --lm-weight "$weight" \
    $nbest_list $am_score_file \
    "$lm_score_file" > "$result_file"
    echo LM scale "$weight" >> results-"$base".txt
    . compute_wer.sh "$result_file" | tee --append results-"$base".txt
done
