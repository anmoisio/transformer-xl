#!/bin/bash -e

module load kaldi-vanilla
module list
export PYTHONIOENCODING='utf-8'

nbest_dir="/scratch/work/moisioa3/conv_lm/nbest"
gen_dir="/scratch/work/moisioa3/conv_lm/transformer-xl/pytorch/generated/lstm-lats"
rescore_dir="/scratch/work/moisioa3/conv_lm/transformer-xl/data/rescored-lstm"

mkdir -p $gen_dir

# model="20200811-133946-55045847"
# model="20200811-225532-55059106"
# model="20200823-103815-55144064"
# model="20200826-132343-55162922"
model="lstm"
n=50
morph="-morph"
# morph=""

results () {
  local test_set="${1}"

  # "${nbest_dir}/${test_set}/chain-${n}best${morph}/text" \
  # "${nbest_dir}/${test_set}/chain-${n}best${morph}/ac_cost" \
  # "${rescore_dir}/${test_set}-${n}-best${morph}/text-${model}"

  bash results.sh \
  "${nbest_dir}/${test_set}/morph-5-gram-lstm-rescored-lats-to-50best/text" \
  "${nbest_dir}/${test_set}/morph-5-gram-lstm-rescored-lats-to-50best/ac_cost" \
  "${nbest_dir}/${test_set}/morph-5-gram-lstm-rescored-lats-to-50best/lm_cost" \
  ${gen_dir}/${test_set}
  # "${rescore_dir}/${test_set}-${n}-best${morph}-lstm-rescored-lats/text-${model}" \

  for filename in ${gen_dir}/${test_set}/hypoth_lm_cost_lms{?,??}.txt
  do
    echo combine segmented text in "${filename}" write to "${filename%.txt}"-combined.txt
    /scratch/work/moisioa3/conv_lm/scripts/combine.py --id2end --input-trn "${filename}" --output-trn "${filename%.txt}"-${n}best-combined.txt
  done

  /scratch/work/moisioa3/conv_lm/scripts/score.sh ${test_set} ${gen_dir}/${test_set}/hypoth_lm_cost_lms{8,9,10,11,12}-${n}best-combined.txt > results/lst-lats/results-${model}-${test_set}-${n}best.txt
}

results devel
# results eval