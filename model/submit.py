import logging
import argparse
import json
import numpy as np
import polars as pl
import pandas as pd
import glob
import os.path
from tqdm import tqdm
from typing import List, Dict, Union

import config
from utils import set_display_options

set_display_options()
log = logging.getLogger(os.path.basename(__file__))


def get_last_commit_hash():
    try:
        import subprocess
        result = subprocess.check_output(['git', 'log', '-1', '--pretty=format:"%H"'])
        return result.decode('utf-8').replace('"', '')[:8]
    except Exception as e:
        return None


def get_timestamp():
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d%H%M%S")


def get_submit_file_name(tag=None):
    tag = '' if args.tag is None else f'-{tag}'
    commit_hash = '' if get_last_commit_hash() is None else f'-{get_last_commit_hash()}'
    timestamp = f'-{get_timestamp()}'
    return f'submission{tag}{commit_hash}{timestamp}'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_split_alias', default='train-test')
    parser.add_argument('--keep_top_k', type=int, default=20)
    parser.add_argument('--tag', default=f'v{config.VERSION}')
    args = parser.parse_args()
    # python -m model.submit --data_split_alias full

    log.info(f'Running {os.path.basename(__file__)} with parameters: \n' + json.dumps(vars(args), indent=2))
    log.info('This generates a submission for kaggle.')

    dir_ranked = f'{config.DIR_DATA}/{args.data_split_alias}-ranked'
    dir_out = f'{config.DIR_DATA}/{args.data_split_alias}-submit'
    os.makedirs(dir_out, exist_ok=True)

    df_submit = []
    recalls = {}

    for target in ['clicks', 'carts', 'orders']:
        log.debug(f'target={target}')
        df_pred = pl.read_parquet(f'{dir_ranked}/{target}/*.parquet')

        if 'pred_rank' in df_pred.columns:
            df_pred = df_pred.drop('pred_rank')

        if 'aid' in df_pred.columns:
            df_pred = df_pred.rename({'aid': 'aid_next'})

        if 'is_retrieved' not in df_pred.columns:
            df_pred = df_pred.with_column(pl.lit(1).alias('is_retrieved'))

        df_pred = df_pred\
            .filter(pl.col('is_retrieved') == 1)\
            .with_column(pl.col('pred_score').rank('ordinal', reverse=True)
                         .over('session').cast(pl.Int16).alias('pred_rank'))\
            .filter((pl.col('pred_rank') <= args.keep_top_k)) \
            .sort(['session', 'pred_rank'])

        log.debug('\n' + str(df_pred.head(10)))

        df_pred = df_pred.with_column(pl.lit(target).alias('type')) \
            .with_column(pl.format('{}_{}', 'session', 'type').alias('session_type')) \
            .groupby('session_type') \
            .agg([pl.col('aid_next').cast(str).list().alias('labels')]) \
            .with_column(pl.col('labels').arr.join(' '))\
            .sort('session_type')

        df_submit.append(df_pred)

    df_submit = pl.concat(df_submit).sort('session_type')

    file_out = f'{dir_out}/{get_submit_file_name(args.tag)}.csv'
    df_submit.write_csv(file_out, has_header=True)
    log.info(f'submission saved to {file_out}')
