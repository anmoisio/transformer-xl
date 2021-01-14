#!/bin/bash -e

module load kaldi-vanilla
module list
export PYTHONIOENCODING='utf-8'

project=/scratch/work/moisioa3/conv_lm
path=${project}/experiments/theanolm-morph-42k
am=tdnn7q_sp_ensemble2
ngram=_morph_nosp_4-gram
lstm=expt2-sampled-seq40

gen_dir="${project}/transformer-xl/pytorch/generated/${am}${ngram}_${lstm}"

mkdir -p $gen_dir

model="20200826-132343-55162922"
n=50

results () {
  local test_set="${1}"

  nbest_dir=${path}/${lstm}/lstm_rescored_${n}best_${test_set}_${am}${ngram}

  bash results.sh \
  "${nbest_dir}/text" \
  "${nbest_dir}/ac_cost" \
  "${nbest_dir}/lm_cost" \
  ${gen_dir}/${test_set}

  for filename in ${gen_dir}/${test_set}/hypoth_lm_cost_lms{?,??}.txt
  do
    echo combine segmented text in "${filename}" write to "${filename%.txt}"-combined.txt
    ${project}/scripts/combine.py --id2end --input-trn "${filename}" --output-trn "${filename%.txt}"-${n}best-combined.txt
  done

  ${project}/scripts/score.sh ${test_set} ${gen_dir}/${test_set}/hypoth_lm_cost_lms{8,9,10,11,12}-${n}best-combined.txt > results/lst-lats/results-${model}-${test_set}-${am}${ngram}_${lstm}-${n}best.txt
}

results devel
results eval