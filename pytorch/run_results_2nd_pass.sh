#!/bin/bash -e

module load kaldi-vanilla
module list
export PYTHONIOENCODING='utf-8'


am_path=/scratch/work/moisioa3/keskustelu2020/experiments/am/converse_fin
am_dir=${am_path}/exp/chain
am=tdnn7q_sp_ensemble2
ngram=_morph_nosp_4-gram

gen_dir="/scratch/work/moisioa3/conv_lm/transformer-xl/pytorch/generated/${am}${ngram}"
rescore_dir="/scratch/work/moisioa3/conv_lm/transformer-xl/data/rescored/"

mkdir -p $gen_dir

model="20200826-132343-55162922"
n=50

results () {
  local test_set="${1}"

  nbest_dir=${am_dir}/${am}/${n}best_${test_set}${ngram}

  bash results.sh \
  "${nbest_dir}/text" \
  "${nbest_dir}/ac_cost" \
  "${nbest_dir}/lm_cost" \
  ${gen_dir}/${test_set}
  # "${rescore_dir}/${test_set}-${n}-best_${am}${ngram}/text-${model}" \

  for filename in ${gen_dir}/${test_set}/hypoth_lm_cost_lms{?,??}.txt
  do
    echo combine segmented text in "${filename}" write to "${filename%.txt}"-combined.txt
    /scratch/work/moisioa3/conv_lm/scripts/combine.py --id2end --input-trn "${filename}" --output-trn "${filename%.txt}"-${n}best-combined.txt
  done

  /scratch/work/moisioa3/conv_lm/scripts/score.sh ${test_set} ${gen_dir}/${test_set}/hypoth_lm_cost_lms{8,9,10,11,12}-${n}best-combined.txt > results/lst-lats/results-${model}-${test_set}-${am}${ngram}-${n}best.txt
}

results devel
results eval