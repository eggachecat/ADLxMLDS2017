#!/usr/bin/env bash
#!/usr/bin/env bash
if [ ! -d "./models/" ]; then
    wget -O rnn.zip "https://www.dropbox.com/s/1cy7h6urtoy4po0/rnn.rar?dl=1"
    unzip rnn.zip
fi


python model_rnn.py -a 4 -dp $1 -tp $2 -mp ./rnn/