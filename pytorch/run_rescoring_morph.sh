#!/bin/bash -e
#SBATCH --mem=18G
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/work/moisioa3/conv_lm/transformer-xl/pytorch/slurm-output/%x-%j.out

. ./path.sh
module list

model_path="/scratch/work/moisioa3/conv_lm/transformer-xl/pytorch"
dataset="wdtrain-morph"
model_folder="LM-TFM-${dataset}"
model="20200826-132343-55162922"
data_dir="../data/web-dsp-morph-42k/"
temp_file=$(mktemp tmp/rescore.XXXXXX)
test_sets="devel eval"
n=50

am=/scratch/work/moisioa3/keskustelu2020/experiments/am/converse_fin/exp/chain/tdnn7q_sp_ensemble2
ngram=_morph_nosp_4-gram

for test_set in ${test_sets}
do
  n_best_file=${am}/50best_${test_set}${ngram}/text
  out_dir="../data/rescored/${test_set}-${n}-best${ngram}/"
  mkdir -p "${out_dir}"

  python3 rescore.py --cuda \
    --data "${data_dir}" \
    --dataset "${dataset}" \
    --tmp "${temp_file}" \
    --out-dir "${out_dir}" \
    --nbest-file "${n_best_file}" \
    --model "${model}" \
    --work-dir "${model_path}/${model_folder}" \

  rm $temp_file
done