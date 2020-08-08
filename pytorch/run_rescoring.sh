#!/bin/bash -e
#SBATCH --mem=18G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1

. ./path.sh
module list
export PYTHONIOENCODING='utf-8'
models="20200807-113927"
temp_file=$(mktemp tmp/rescore.XXXXXX)

python3 rescore.py \
  --data ../data/web-dsp/ \
  --tmp $temp_file \
  --out-dir ../data/results/ \
  --nbest-file /scratch/work/moisioa3/conv_lm/nbest/devel/chain-50best/text \
  --models "$models" \
  --cuda \
  --work-dir /scratch/work/moisioa3/conv_lm/transformer-xl/pytorch/LM-TFM-wdtrain/ \

rm $temp_file