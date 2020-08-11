#!/bin/bash -e

module load kaldi-vanilla

bash results.sh \
  /scratch/work/moisioa3/conv_lm/nbest/devel/chain-1000best/text \
  /scratch/work/moisioa3/conv_lm/nbest/devel/chain-1000best/ac_cost \
  /scratch/work/moisioa3/conv_lm/transformer-xl/data/results/1000best/text-20200807-113927

