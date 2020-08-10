#!/bin/bash -e
#SBATCH --mem=18G
#SBATCH --time=1-00
#SBATCH --gres=gpu:1

. ./path.sh
module list
export PYTHONIOENCODING='utf-8'

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train.py \
        --cuda \
        --data ../data/web-dsp/ \
        --dataset wdtrain \
        --n_layer 32 \
        --d_model 256 \
        --n_head 8 \
        --d_head 40 \
        --d_inner 1024 \
        --dropout 0.2 \
        --dropatt 0.2 \
        --optim adam \
        --lr 0.0001 \
        --max_step 1200000 \
        --batch_chunk 4 \
        --tgt_len 32 \
        --mem_len 32 \
        --eval_tgt_len 32 \
        --batch_size 256 \
        --gpu0_bsz -1 \
        --restart \
        --restart_dir /scratch/work/moisioa3/conv_lm/transformer-xl/pytorch/LM-TFM-wdtrain/20200808-233619/ \
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python eval.py \
        --cuda \
        --data ../data/enwik8/ \
        --dataset enwik8 \
        --tgt_len 80 \
        --mem_len 2100 \
        --clamp_len 820 \
        --same_length \
        --split test \
        ${@:2}
else
    echo 'unknown argument 1'
fi
