#!/usr/bin/env bash

# $1 -> the data directory,
# $2 -> test data output filename
# $3 -> peer review output filename


python main.py -a 1  -dp $1 -op $2 -mp ./models/model-basic-emb-25/model.ckpt
python main.py -a 4  -dp $1 -op $3 -mp ./models/model-basic-emb-25/model.ckpt