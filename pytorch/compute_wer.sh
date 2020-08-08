#!/bin/bash

filter='sed -e "s:+ +::g" -e "s: +::g" -e "s:+ ::g" -e "s:<UNK>::g"'
ref=/scratch/work/moisioa3/conv_lm/data/devel/verbatim.ref
use_bootci=false
compare=

. ./path.sh
. ../parse_options.sh

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <hyp.txt>"
  exit 1
fi

hypfile="$1"


if [ "$use_bootci" == "false" ]; then
  compute-wer --mode="present" ark:"$ref" ark:"$filter <$hypfile|"
else
  if [ -z "$compare" ]; then 
    compute-wer-bootci --mode="present" ark:"$ref" ark:"$filter <$hypfile|"
  else
    compute-wer-bootci --mode="present" ark:"$ref" ark:"$filter <$hypfile|" ark:"$filter <$compare|"
  fi
fi
