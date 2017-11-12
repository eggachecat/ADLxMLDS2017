#!/usr/bin/env bash

$1 # the data directory,
$2 # test data output filename
$3 # peer review output filename


python main.py -a 2  -dp $1 -a $1 -op $3 -mp ./models/model.ckpt