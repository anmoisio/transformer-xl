# coding: utf-8
import argparse
import time
import math
import os, sys
import tempfile

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from data_utils import get_lm_corpus
from mem_transformer import MemTransformerLM
from utils.exp_utils import get_logger

parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
parser.add_argument('--data', type=str, default='../data/web-dsp/',
                    help='location of the data corpus')
parser.add_argument('--tmp', type=str, default='tmp/t',
                    help='location of the temporary file used in this tool')
parser.add_argument('--out-dir', type=str, default='generated',
                    help='location of the output directory')
parser.add_argument('--nbest-file', type=str, default='../../nbest/devel/chain-200best-morph/text',
                    help='location of the nbest file')
parser.add_argument('--models', type=str, default='20191112-102012 20191022-134318',
                    help='list of model ids to be used for rescoring')
parser.add_argument('--sent-sep', type=str, default='<S>',
                    help='sentence separator symbol')
parser.add_argument('--dataset', type=str, default='wdtrain',
                    choices=['Ktrain', 'wdtrain', 'wdtrain-morph'],
                    help='dataset name')
parser.add_argument('--ext-len', type=int, default=0,
                    help='length of the extended context')
parser.add_argument('--mem-len', type=int, default=0,
                    help='length of the retained previous heads')
parser.add_argument('--clamp-len', type=int, default=-1,
                    help='max positional embedding index')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--work-dir', type=str, default='./',
                    help='path to the work_dir')
parser.add_argument('--same-length', action='store_true',
                    help='set same length attention with masking')
args = parser.parse_args()

assert args.ext_len >= 0, 'extended context length must be non-negative'

if torch.cuda.is_available():
    if not args.cuda:
        print('WARNING: You have a CUDA device, so you should probably run with --cuda')

device = torch.device('cuda' if args.cuda else 'cpu')

# Load dataset
all_ids = []
space_counter = 0
tmpfile = open(args.tmp, 'w', encoding='utf-8')
with open(args.nbest_file, "r", encoding="utf-8") as reader:
    while True:
        line = reader.readline()
        if not line:
            break

        line = line.strip()
        tokens = line.split(" ", 1)
        if len(tokens) == 1:
            tokens.append(args.sent_sep)

        all_ids.append(tokens[0])
        tmpfile.write(tokens[1] + '\n')

tmpfile.close()


def rescore(corpus, nbest_file, model, ext_len, mem_len, outfile):
    encoded_sent = corpus.vocab.encode_file(path=nbest_file, add_double_eos=True)
    for idx, sent in enumerate(encoded_sent):
        streams = [None] * 1
        bptt = len(list(sent)) - 1
        data = torch.LongTensor(bptt, 1)
        target = torch.LongTensor(bptt, 1)
        model.reset_length(bptt, ext_len, mem_len)
        n_retain = 0
           
        # data   : [n_retain+bptt x bsz]
        # target : [bptt x bsz]
        data[n_retain:].fill_(-1)
        target.fill_(-1)
        for i in range(1):
            n_filled = 0
            while n_filled < bptt:
                if streams[i] is None or len(streams[i]) <= 1:
                    streams[i] = sent
                # number of new tokens to fill in
                n_new = min(len(streams[i]) - 1, bptt - n_filled)
                # first n_retain tokens are retained from last batch
                data[n_retain + n_filled : n_retain + n_filled + n_new, i] = \
                    streams[i][:n_new]
                target[n_filled: n_filled + n_new, i] = streams[i][1: n_new + 1]
                streams[i] = streams[i][n_new:]
                n_filled += n_new

        data = data.to(device)
        target = target.to(device)
        model.eval()
        mems = tuple()
        with torch.no_grad():
            ret = model(data, target, *mems)
            loss = ret[0]
            loss = loss.sum()
            sent_rescore = ''
            sent_rescore = all_ids[idx] + ' ' + str(loss.item())
            outfile.write(sent_rescore + '\n')


corpus = get_lm_corpus(args.data, args.dataset)
model_dirs = args.models.split()
for model_dir in model_dirs:
    # Load the best saved model.
    model = None
    with open(os.path.join(args.work_dir, model_dir, 'model.pt'), 'rb') as f:
        model = torch.load(f)

    model.backward_compatible()
    model = model.to(device)

    if args.clamp_len > 0:
        model.clamp_len = args.clamp_len
        
    if args.same_length:
        model.same_length = True
    
    with open(os.path.join(args.out_dir, os.path.basename(args.nbest_file) + '-' + model_dir), 'w', encoding='utf-8') as outfile:
        rescore(corpus, args.nbest_file, model, args.ext_len, args.mem_len, outfile)

