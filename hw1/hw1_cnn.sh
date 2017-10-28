#!/usr/bin/env bash
if [ ! -d "./models/" ]; then
    wget -O models.zip "https://www.dropbox.com/s/ha9ih272gzjb9x9/models.zip?dl=1"
    unzip models.zip
fi


python model_cnn.py -a 1 -dp $1 -tp $2 -mp ./models/cnn/weights.27-0.33732.hdf5