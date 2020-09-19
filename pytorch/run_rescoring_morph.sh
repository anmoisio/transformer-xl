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
test_set=devel
nbest_dir="/scratch/work/moisioa3/conv_lm/nbest"
n=50
n_best_file="${nbest_dir}/${test_set}/morph-5-gram-lstm-${n}best/text"
out_dir="../data/rescored-lstm/${test_set}-${n}-best-morph/"
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