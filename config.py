# ******************************************************************************
# This contains all configs/parameters used in this project.
# ******************************************************************************

from pathlib import Path
import os
import json
import copy
import logging
from typing import List

VERSION = '1.0.0'

# Directories
# ******************************************************************************
DIR_PROJ = (Path(__file__) / '..').resolve()
DIR_DATA = f'{DIR_PROJ}/data'
DIR_ARTIFACTS = f'{DIR_PROJ}/artifacts'
os.makedirs(DIR_ARTIFACTS, exist_ok=True)  # Create dir for artifacts if it does not exist

# Logging
# ******************************************************************************
LOGS_LEVEL = logging.DEBUG
FILE_LOGS = f'{DIR_ARTIFACTS}/logs.log'
# set logging config:
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(FILE_LOGS), logging.StreamHandler()],
    level=LOGS_LEVEL,
)

# Feature engineering
# ******************************************************************************
TYPES = ['clicks', 'carts', 'orders']
TYPE2ID = {'clicks': 0, 'carts': 1, 'orders': 2}
ID2TYPE = {0: 'clicks', 1: 'carts', 2: 'orders'}
# df['type'].map(type2id) # df['type'].map(lambda i: id2type[i])

# filter co-events when doing self merge
MIN_TIME_TO_NEXT = -24 * 60 * 60  # value zero means that next event can't be before this event
MAX_TIME_TO_NEXT = 24 * 60 * 60  # 24 hours * 60 min * 60 sec
MAP_MAX_TIME_TO_NEXT = {
    'click_to_click': 12 * 60 * 60,  # 12 hours * 60 min * 60 sec
    'click_to_cart_or_buy': MAX_TIME_TO_NEXT,
    'cart_to_cart': MAX_TIME_TO_NEXT,
    'cart_to_buy': MAX_TIME_TO_NEXT,
    'buy_to_buy': MAX_TIME_TO_NEXT,
}


OPTIM_ROWS_POLARS_GROUPBY = 100_000_000
MAX_ROWS_POLARS_GROUPBY = 300_000_000

# minimum count to be saved on disk
MIN_COUNT_TO_SAVE = {
    'click_to_click': 10,
    'click_to_cart_or_buy': 5,
    'cart_to_cart': 2,
    'cart_to_buy': 2,
    'buy_to_buy': 2,
}
MIN_COUNT_IN_PART = {'click_to_click': 2, 'click_to_cart_or_buy': 2}
MAX_CO_EVENT_PAIRS_TO_SAVE_DISK = 300_000_000

# which counts to compute
CO_EVENTS_TO_COUNT = [
    'click_to_click',
    'click_to_cart_or_buy',
    'cart_to_cart',
    'cart_to_buy',
    'buy_to_buy',
]

# RETRIEVAL
# to retrieve candidates co-events, keep only the last N events in session
# use a high number to keep all
RETRIEVE_N_LAST_CLICKS = 99  # 30: percentile 99%
RETRIEVE_N_LAST_CARTS = 99  # 25: percentile 99.5%
RETRIEVE_N_LAST_ORDERS = 99  # 25: percentile 99.5%
RETRIEVE_N_MOST_FREQUENT = 99  #

MAP_NAME_COUNT_TYPE = {
    # (event type to next event type(s))
    'click_to_click': (0, [0]),
    'click_to_cart_or_buy': (0, [1, 2]),
    'cart_to_cart': (1, [1]),
    'cart_to_buy': (1, [2]),
    'buy_to_buy': (2, [2]),
}

RETRIEVAL_FIRST_N_CO_COUNTS = {
    'click_to_click': 10,
    'click_to_cart_or_buy': 10,
    'cart_to_cart': 20,
    'cart_to_buy': 20,
    'buy_to_buy': 20,
}

RETRIEVAL_CO_COUNTS_TO_JOIN = [
    'click_to_click',
    'click_to_cart_or_buy',
    'cart_to_cart',
    'cart_to_buy',
    'buy_to_buy',
]

W2VEC_USE_CACHE = True
W2VEC_MODELS = {
    'word2vec-train-test-types-all-size-100-mincount-5-window-10': {
        # source of sessions (as sentences) with AIDs (as words)
        'dir_sessions': [
            f'{DIR_DATA}/train-test-parquet/train_sessions/*.parquet',
            f'{DIR_DATA}/train-test-parquet/test_sessions/*.parquet'
        ],
        'types': [0, 1, 2],  # which event types to filter
        # word2vec embedding parameters:
        'params': {
            'vector_size': 100,
            'window': 10,
            'min_count': 5,
        },
        'k': 20,  # number of neighbours to retrieve
        'first_n_aids': 600_000,  # for how many AIDs (words) to find neighbours (output df has first_n_aids*k rows)
        # params for faiss index:
        'nlist': 100, # how many cells
        'nprobe': 3,  # how many closest cells to search
        # params for annoy index:
        'n_trees': 20,  # number of trees
    },
    'word2vec-train-test-types-1-2-size-100-mincount-5-window-10': {
        # source of sessions (as sentences) with AIDs (as words)
        'dir_sessions': [
            f'{DIR_DATA}/train-test-parquet/train_sessions/*.parquet',
            f'{DIR_DATA}/train-test-parquet/test_sessions/*.parquet'
        ],
        'types': [1, 2],  # which event types to filter
        # word2vec embedding parameters:
        'params': {
            'vector_size': 100,
            'window': 10,
            'min_count': 5,
        },
        'k': 20,  # number of neighbours to retrieve
        'first_n_aids': 600_000,  # for how many AIDs (words) to find neighbours (output df has first_n_aids*k rows)
        # params for faiss index:
        'nlist': 100,  # how many cells
        'nprobe': 3,  # how many closest cells to search
    },
    'word2vec-full-types-all-size-100-mincount-5-window-10': {
        # source of sessions (as sentences) with AIDs (as words)
        'dir_sessions': [
            f'{DIR_DATA}/full-parquet/train_sessions/*.parquet',
            f'{DIR_DATA}/full-parquet/test_sessions/*.parquet'
        ],
        'types': [0, 1, 2],  # which event types to filter
        # word2vec embedding parameters:
        'params': {
            'vector_size': 100,
            'window': 10,
            'min_count': 5,
        },
        'k': 20,  # number of neighbours to retrieve
        'first_n_aids': 600_000,  # for how many AIDs (words) to find neighbours (output df has first_n_aids*k rows)
        # params for faiss index:
        'nlist': 100, # how many cells
        'nprobe': 3,  # how many closest cells to search
        # params for annoy index:
        'n_trees': 20,  # number of trees
    },
    'word2vec-full-types-1-2-size-100-mincount-5-window-10': {
        # source of sessions (as sentences) with AIDs (as words)
        'dir_sessions': [
            f'{DIR_DATA}/full-parquet/train_sessions/*.parquet',
            f'{DIR_DATA}/full-parquet/test_sessions/*.parquet'
        ],
        'types': [1, 2],  # which event types to filter
        # word2vec embedding parameters:
        'params': {
            'vector_size': 100,
            'window': 10,
            'min_count': 5,
        },
        'k': 20,  # number of neighbours to retrieve
        'first_n_aids': 600_000,  # for how many AIDs (words) to find neighbours (output df has first_n_aids*k rows)
        # params for faiss index:
        'nlist': 100,  # how many cells
        'nprobe': 3,  # how many closest cells to search
    },
}
