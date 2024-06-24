#!/bin/sh

# install requirements:
pip install -r "./requirements.txt"

# download bart score:
wget -P "./resources/bart_score.py" "https://github.com/neulab/BARTScore/blob/main/bart_score.py"

# download and process food incidents dataset:
wget -P "./data/food incidents - hazard/food_recall_incidents.csv" "https://zenodo.org/records/10891602/files/food_recall_incidents.csv?download=1"
python "./data/food incidents - hazard/sample.py"

# download and process movie dataset:
wget -P "./data/movies/food_recall_incidents.csv" -q -O - https://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz | tar xvzf -
python "data/movies/sample.py"