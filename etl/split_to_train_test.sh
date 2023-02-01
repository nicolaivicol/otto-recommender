#!/bin/bash
# this script creates a train/test split
echo "WARNING: script testset has ETA of ~40 min"
cd ../../recsys-dataset || exit
echo $(pwd)
python -m src.testset \
  --train-set ../otto-recommender/data/full/train_sessions.jsonl \
  --days 7 \
  --output-path ../otto-recommender/data/train-test/
