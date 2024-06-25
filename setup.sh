#!/bin/sh

# install requirements:
pip install -r ./requirements.txt

# download bart score:
wget -P ./resources/ https://raw.githubusercontent.com/neulab/BARTScore/main/bart_score.py

cd ./data

# download and process food incidents dataset:
cd "./food incidents - hazard"
wget https://zenodo.org/records/10891602/files/food_recall_incidents.csv
python sample.py

cd ..

# download and process movie dataset:
wget -q -O - http://www.eraserbenchmark.com/zipped/movies.tar.gz | tar xvzf -
cd ./movies
python sample.py