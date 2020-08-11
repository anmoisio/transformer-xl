#!/bin/bash -e
#SBATCH --mem=18G
#SBATCH --time=2-00
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/work/moisioa3/conv_lm/transformer-xl/pytorch/slurm-output/slurm-%j.out

. ./path.sh
module list
export PYTHONIOENCODING='utf-8'
models="20200807-113927"
temp_file=$(mktemp tmp/rescore.XXXXXX)

python3 rescore.py \
  --data ../data/web-dsp/ \
  --tmp $temp_file \
  --out-dir ../data/results/1000best/ \
  --nbest-file /scratch/work/moisioa3/conv_lm/nbest/devel/chain-1000best/text \
  --models "$models" \
  --cuda \
  --work-dir /scratch/work/moisioa3/conv_lm/transformer-xl/pytorch/LM-TFM-wdtrain/ \

rm $temp_file