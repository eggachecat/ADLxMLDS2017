#!/usr/bin/env bash

source ~/workspace/env/bin/activate
python main.py -a $1 -dp $2 -op $3 -mp ./outputs/model.ckpt