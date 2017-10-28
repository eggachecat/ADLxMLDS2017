#!/usr/bin/env bash
if [ ! -d "./models/" ]; then
    wget -O models.zip "https://www.dropbox.com/s/ha9ih272gzjb9x9/models.zip?dl=1"
    unzip models.zip
fi

python model_rnn.py -a 4 -dp $1 -tp $2 -mp ./models/rnn/
