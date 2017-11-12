#!/usr/bin/env bash

$1 # the data directory,
$2 # test data output filename


python main.py -a 2  -dp $1 -a $1 -op $2 -mp ./models/model.ckpt