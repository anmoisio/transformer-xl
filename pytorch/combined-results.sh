#!/bin/bash -e

module purge
module load sctk
module list

model="20200806-163347"
gen_dir="/scratch/work/moisioa3/conv_lm/transformer-xl/pytorch/generated"

results () {
    local test_set="${1}"
    /scratch/work/moisioa3/conv_lm/scripts/score.sh ${test_set} ${gen_dir}/hypoth_text-${model}_lms{8,9,10,11,12}-combined.??? > results-${model}-${test_set}.txt
}

results devel
# results eval