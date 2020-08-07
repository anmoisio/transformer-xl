#!/bin/bash -e
#SBATCH --mem=18G
#SBATCH --time=4:00:00
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
        --n_layer 72 \
        --d_model 256 \
        --n_head 8 \
        --d_head 40 \
        --d_inner 1024 \
        --dropout 0.05 \
        --dropatt 0.05 \
        --optim adam \
        --lr 0.00025 \
        --warmup_step 40000 \
        --max_step 1200000 \
        --batch_chunk 4 \
        --tgt_len 32 \
        --mem_len 32 \
        --eval_tgt_len 32 \
        --batch_size 512 \
        --gpu0_bsz -1 \
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
