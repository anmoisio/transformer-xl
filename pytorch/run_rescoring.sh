#!/bin/bash -ex
#SBATCH --mem=18G
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/work/moisioa3/conv_lm/transformer-xl/pytorch/slurm-output/%x-%j.out

. ./path.sh
module list

model_path="/scratch/work/moisioa3/conv_lm/transformer-xl/pytorch"
dataset="wdtrain"
model_folder="LM-TFM-${dataset}"
model="20200811-133946-55045847"
data_dir="../data/web-dsp/"
temp_file=$(mktemp tmp/rescore.XXXXXX)
test_set=devel
nbest_dir="/scratch/work/moisioa3/conv_lm/nbest"
n=50
n_best_file="${nbest_dir}/${test_set}/chain-${n}best/text"
out_dir="../data/rescored/${test_set}-${n}-best/"
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