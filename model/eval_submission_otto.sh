#!/bin/bash
# this script evaluates a submission using the script provided by the organizer (Otto)
cd ../../recsys-dataset || exit
echo $(pwd)
python -m src.evaluate \
  --test-labels ../otto-recommender/data/train-test/test_labels.jsonl \
  --predictions ../otto-recommender/data/train-test-submit/submission-v1.0.0-7fa08333-20230119141547.csv
