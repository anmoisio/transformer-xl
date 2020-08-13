#!/bin/bash

model="20200811-225532-55059106"
model="20200806-163347"
model="20200811-133946-55045847"

for test_set in devel
do
    for filename in /scratch/work/moisioa3/conv_lm/transformer-xl/pytorch/generated/hypoth_text-${model}_lms{?,??}.txt
    do
        echo combine segmented text in "${filename}" write to "${filename%.trn}"-combined.trn
        /scratch/work/moisioa3/conv_lm/scripts/combine.py --id2end --input-trn "${filename}" --output-trn "${filename%.txt}"-combined.trn
    done
done