#!/usr/bin/env python
# coding=utf-8

# convert theanoLM nbest files to kaldi format so they can be used by kaldi-utensils/cutlery/rescore_nbest.py

import argparse
import os.path

parser = argparse.ArgumentParser(description='Parse TheanoLM .nnlm-probs file')
parser.add_argument('--nbest-file', type=str, default='test.nnlm-probs',
                    help='the input nbest file')
parser.add_argument('--rescored-file', type=str, default='text',
                    help='the output nbest file')
parser.add_argument('--lm-cost-file', type=str, default='lm_cost',
                    help='output lm-cost file')
parser.add_argument('--am-cost-file', type=str, default='am_cost',
                    help='output am-cost file')
args = parser.parse_args()

print(args.nbest_file)

if os.path.isfile(args.am_cost_file):
    raise ValueError('am cost file exists - will not overwrite')
if os.path.isfile(args.lm_cost_file): 
    raise ValueError('lm cost file exists - will not overwrite')
if os.path.isfile(args.rescored_file): 
    raise ValueError('text file exists - will not overwrite')

lm_cost=open(args.lm_cost_file, 'w', encoding='utf-8')
am_cost=open(args.am_cost_file, 'w', encoding='utf-8')
rescored_file=open(args.rescored_file, 'w', encoding='utf-8')
utt_id = ''
hyp_num = 0
with open(args.nbest_file, 'r', encoding='utf-8') as f:
    while True: 
        line = f.readline()
        if not line: break

        # ascore lscore nwords w1 w2 w3 ...
        split_line = line.split(' ', 4)

        if utt_id == split_line[0]:
            hyp_num += 1
        else:
            hyp_num = 1
        utt_id = split_line[0]

        rescored_file.write(split_line[0] + '-' + str(hyp_num) + ' ' + split_line[4] )
        am_cost.write(split_line[0] + '-' + str(hyp_num) + ' ' + split_line[1] + '\n')
        lm_cost.write(split_line[0] + '-' + str(hyp_num) + ' ' + split_line[2] + '\n')

