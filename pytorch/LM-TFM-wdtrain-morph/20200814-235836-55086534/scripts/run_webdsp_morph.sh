#!/bin/bash -e
#SBATCH --mem=18G
#SBATCH --time=2-00
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/work/moisioa3/conv_lm/transformer-xl/pytorch/slurm-output/%x-%j.out

. ./path.sh
export PYTHONIOENCODING='utf-8'
module list

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train.py \
        --cuda \
        --data ../data/web-dsp-morph-42k/ \
        --dataset wdtrain-morph \
        --n_layer 32 \
        --d_model 256 \
        --n_head 8 \
        --d_head 40 \
        --d_inner 1024 \
        --dropout 0.1 \
        --dropatt 0.1 \
        --optim adam \
        --lr 0.0001 \
        --warmup_step 20000 \
        --max_step 120000 \
        --batch_chunk 1 \
        --tgt_len 32 \
        --mem_len 32 \
        --eval_tgt_len 32 \
        --batch_size 64 \
        --gpu0_bsz -1 \
        --job_id "${SLURM_JOB_ID}" \
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
