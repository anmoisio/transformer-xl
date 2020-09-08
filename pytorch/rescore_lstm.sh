#!/bin/bash

./get_LSTM_50_from_nbest.py \
    /scratch/work/moisioa3/conv_lm/transformer-xl/data/rescored/devel-50-best/text-20200811-133946-55045847 \
    /scratch/work/moisioa3/conv_lm/nbest/devel/chain-50best/ac_cost \
    /scratch/work/moisioa3/conv_lm/nbest/devel/chain-50best/lm_cost \
    --lm-weight 10