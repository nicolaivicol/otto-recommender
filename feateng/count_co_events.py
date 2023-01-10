import math
import time
import polars as pl
from typing import Dict, List
from tqdm.auto import tqdm
import json
import os
import glob
import logging
from pathlib import Path
import argparse
import config

log = logging.getLogger()


def self_merge(df_part: pl.DataFrame):
    # self merge by session, have an outer product by aid within each session
    df_part_merged = df_part.join(df_part, on='session', suffix='_next')
    # 0.8 sec on 5000 sessions, polars is 4x faster than pandas

    # remove rows where same event is joined by itself
    df_part_merged = df_part_merged.filter(
        ~((pl.col('aid') == pl.col('aid_next'))
          & (pl.col('ts') == pl.col('ts_next'))
          & (pl.col('type') == pl.col('type_next')))
    )

    # add column time to next (this will also be used afterwards for further filters)
    df_part_merged = df_part_merged.with_columns([(pl.col('ts_next') - pl.col('ts')).alias('time_to_next')])

    # filter by time between events  (0.7 sec)
    df_part_merged = df_part_merged.filter(
        (pl.col('time_to_next') >= config.MIN_TIME_TO_NEXT)
        & (pl.col('time_to_next') <= config.MAX_TIME_TO_NEXT)
    )

    return df_part_merged


def self_merge_big_df(df: pl.DataFrame, n_sessions_in_part = 10000):
    # iterate over parts, self merge smaller parts
    sessions = df['session'].unique()
    n_sessions = len(sessions)
    n_parts = math.ceil(n_sessions / n_sessions_in_part)

    df_parts_merged = []
    for i_part in range(n_parts):
        i_start_session = i_part * n_sessions_in_part
        i_end_session = min((i_part + 1) * n_sessions_in_part, n_sessions)
        sessions_part = sessions[i_start_session:i_end_session]
        df_part = df.filter(pl.col('session').is_in(sessions_part))
        df_part_merged = self_merge(df_part)
        df_parts_merged.append(df_part_merged)

    df_merge = pl.concat(df_parts_merged)
    return df_merge


def count_co_events(df_merged: pl.DataFrame) -> Dict['str', pl.DataFrame]:
    log.debug(f'compute_co_events(): input to has {df_merged.shape} rows')
    counts_co_events = {}
    co_events_to_count = config.CO_EVENTS_TO_COUNT

    if 'count_click_to_click' in co_events_to_count:
        count_click_to_click = df_merged\
            .filter((pl.col('type') == 0)
                    & (pl.col('type_next') == 0)
                    & (pl.col('time_to_next') <= config.MAX_TIME_TO_NEXT_CLICK_TO_CLICK))\
            .groupby(['aid', 'aid_next'])\
            .agg([pl.col('aid_next').count().alias('count'),])
        counts_co_events['count_click_to_click'] = count_click_to_click

    if 'count_click_to_cart_or_buy' in co_events_to_count:
        count_click_to_cart_or_buy = df_merged\
            .filter((pl.col('type') == 0) & (pl.col('type_next').is_in([1, 2])))\
            .groupby(['aid', 'aid_next'])\
            .agg([pl.col('aid_next').count().alias('count'),])
        counts_co_events['count_click_to_cart_or_buy'] = count_click_to_cart_or_buy

    if 'count_cart_to_buy' in co_events_to_count:
        count_cart_to_buy = df_merged\
            .filter((pl.col('type') == 1) & (pl.col('type_next') == 2))\
            .groupby(['aid', 'aid_next'])\
            .agg([pl.col('aid_next').count().alias('count'),])
        counts_co_events['count_cart_to_buy'] = count_cart_to_buy

    if 'count_cart_to_cart' in co_events_to_count:
        count_cart_to_cart = df_merged\
            .filter((pl.col('type') == 1) & (pl.col('type_next') == 1))\
            .groupby(['aid', 'aid_next'])\
            .agg([pl.col('aid_next').count().alias('count'),])
        counts_co_events['count_cart_to_cart'] = count_cart_to_cart

    if 'count_buy_to_buy' in co_events_to_count:
        count_buy_to_buy = df_merged\
            .filter((pl.col('type') == 2) & (pl.col('type_next') == 2))\
            .groupby(['aid', 'aid_next'])\
            .agg([pl.col('aid_next').count().alias('count'),])
        counts_co_events['count_buy_to_buy'] = count_buy_to_buy

    log.debug(f'compute_co_events(): output size '
              f'{json.dumps({name: df.shape[0] for name, df in counts_co_events.items()})}')
    return counts_co_events


def count_co_events_all_files(dir_sessions, dir_stats, skip_if_exists=True):
    files_parquet = sorted(glob.glob(f'{dir_sessions}/*.parquet'))

    for file_parquet in tqdm(files_parquet, desc="Count co-events", total=len(files_parquet)):
        all_exists = all([os.path.exists(f'{dir_stats}/{name_df}/{Path(file_parquet).stem}.parquet')
                          for name_df in config.CO_EVENTS_TO_COUNT])

        if skip_if_exists and all_exists:
            log.debug(f'skipping {Path(file_parquet).stem}.parquet, counts already exist')
            continue

        df = pl.read_parquet(file_parquet)
        df = df.unique()
        df_merge = self_merge_big_df(df)
        counts_co_events = count_co_events(df_merge)

        # save parts to disk
        for name_df, df in counts_co_events.items():
            file_name_out = f'{dir_stats}/{name_df}/{Path(file_parquet).stem}.parquet'
            os.makedirs(os.path.dirname(file_name_out), exist_ok=True)
            counts_co_events[name_df].write_parquet(file_name_out)


def reduce_size_of_df_with_counts(df_tmp, max_rows_allowed):
    # reduce size and memory usage to be able to do groupby on the concatenated data frame
    min_count = 2
    while df_tmp.shape[0] > max_rows_allowed:
        df_tmp = df_tmp.filter((pl.col('count') >= min_count))
        min_count += 1
    return df_tmp


def concat_files_w_stats(name, dir_stats, files_stats=None):
    rows_chunk = config.OPTIM_ROWS_POLARS_GROUPBY

    if files_stats is not None:
        df = pl.concat([pl.read_parquet(f) for f in files_stats])
    else:
        df = pl.read_parquet(f'{dir_stats}/{name}/*.parquet')

    assert df.columns == ['aid', 'aid_next', 'count']

    # groupby by parts if data frame is too big
    if df.shape[0] > rows_chunk:
        list_df_parts = []
        n_parts = math.ceil(df.shape[0] / rows_chunk)
        for i in range(n_parts):
            df_part = df.slice(i * rows_chunk, rows_chunk)
            df_part = df_part.groupby(['aid', 'aid_next']).sum()
            reduce_size_of_df_with_counts(df_part, config.MAX_ROWS_POLARS_GROUPBY / rows_chunk)
            list_df_parts.append(df_part)

        df = pl.concat(list_df_parts)

    df = df.groupby(['aid', 'aid_next']).sum()
    # df = df.groupby(['aid', 'aid_next']).agg([pl.sum('count').alias('count'), ])
    df = df \
        .filter((pl.col('count') >= config.MIN_COUNT_TO_SAVE.get(name, 1))) \
        .sort(['count'], reverse=True)
    df = df.head(config.MAX_CO_EVENT_PAIRS_TO_SAVE_DISK)
    df.with_column(pl.col('count').cast(pl.Int32))
    df.to_pandas().to_parquet(f'{dir_stats}/{name}.parquet')  # save as panda so PyCharm can glimpse into it


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_sessions_train', default='../data/train-test-parquet/train_sessions')
    parser.add_argument('--dir_sessions_test', default='../data/train-test-parquet/test_sessions')
    parser.add_argument('--dir_stats', default='../data/train-test-counts-co-event')
    parser.add_argument('--count', default=True)
    parser.add_argument('--merge', default=True)
    parser.add_argument('--merge_train_test', default=True)
    args = parser.parse_args()

    folder_sessions_train = Path(args.dir_sessions_train).stem
    folder_sessions_test = Path(args.dir_sessions_test).stem
    tic_start = time.time()

    if args.count:
        log.info('count co-events per parquet file (parts)')
        count_co_events_all_files(args.dir_sessions_train, f'{args.dir_stats}/{folder_sessions_train}')
        count_co_events_all_files(args.dir_sessions_test, f'{args.dir_stats}/{folder_sessions_test}')

    if args.merge:
        log.info('merge parts with counts and re-compute counts')
        for name in config.CO_EVENTS_TO_COUNT:
            concat_files_w_stats(name, f'{args.dir_stats}/{folder_sessions_train}')
            concat_files_w_stats(name, f'{args.dir_stats}/{folder_sessions_test}')

    if args.merge_train_test:
        log.info('merge train and test counts and re-compute counts')
        tic = time.time()
        for name in config.CO_EVENTS_TO_COUNT:
            concat_files_w_stats(
                name=name,
                files_stats=[f'{args.dir_stats}/{folder_sessions_train}/{name}.parquet',
                             f'{args.dir_stats}/{folder_sessions_test}/{name}.parquet'],
                dir_stats=args.dir_stats,
            )
        toc = time.time()
        print('merge - total time elapsed:', toc - tic)




# import dask.dataframe as dd
# df = dd.read_parquet('../data/stats/train_sessions/count_click_to_click/*.parquet')

# import pandas as pd
# import random
#
# df_tmp = pd.read_parquet('../data/train-test-counts-co-event/count_click_to_click.parquet')
# random_id = random.choice(list(df_tmp['aid'].unique()))
# df_tmp.loc[df_tmp['aid'] == random_id]

# tic = time.time()
# ....
# toc = time.time()
# print('self join', toc - tic)
# print('total expected of self join', (toc - tic) * 100000 / n_sessions_in_part)  #

# import random
# df_tmp = df_part.filter(pl.col('type').is_in([1,2])).to_pandas()
# random_session = random.choice(list(df_tmp['session'].unique()))
# df_tmp_session = df_tmp[df_tmp['session'] == random_session].copy()
# df_tmp_session['timestamp'] = pd.to_datetime(df_tmp_session['ts'], unit='s')
# df_tmp_session.head(100)
# df_part_merged = df_part_merged.sort(['session', 'ts'])