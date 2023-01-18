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

log = logging.getLogger(os.path.basename(__file__))


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


def self_merge_big_df(df: pl.DataFrame, n_sessions_in_part: int = 10_000):
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

    for count_type, map_this_next in config.MAP_NAME_COUNT_TYPE.items():
        type_this, types_next = map_this_next
        df_tmp = df_merged \
            .filter((pl.col('type') == type_this)
                    & (pl.col('type_next').is_in(types_next))
                    & (pl.col('time_to_next').abs() <= config.MAP_MAX_TIME_TO_NEXT['count_type'])) \
            .groupby(['aid', 'aid_next']) \
            .agg([pl.col('aid_next').count().alias('count')])
        counts_co_events[count_type] = df_tmp

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
            df.write_parquet(file_name_out)


def concat_files_w_stats(name, dir_stats, files_stats=None):
    log.debug(f'merge and aggregate counts for {name}')

    if files_stats is not None:
        df = pl.concat([pl.read_parquet(f) for f in files_stats])
    else:
        df = pl.read_parquet(f'{dir_stats}/{name}/*.parquet')
        # df = pl.read_parquet(f'{dir_stats}/{name}/00000000_00100000.parquet')

    log.debug(f'loaded {df.shape[0]:,} rows in total')

    # this gives error: Process finished with exit code 137 (interrupted by signal 9: SIGKILL)
    # so I need to do an aggregation by parts and chop some pairs
    # df = pl.scan_parquet(f'{dir_stats}/{name}/*.parquet')\
    #     .groupby(['aid', 'aid_next'])\
    #     .agg([pl.sum('count').alias('count')]) \
    #     .sort('count', reverse=True) \
    #     .head(config.MAX_CO_EVENT_PAIRS_TO_SAVE_DISK)\
    #     .collect()

    assert df.columns == ['aid', 'aid_next', 'count']

    # truncate small counts if table is big
    if 'click_to' in name and df.shape[0] > 100_000_000:
        df = df.filter((pl.col('count') >= config.MIN_COUNT_IN_PART.get(name, 1)))

    # groupby by parts if data frame is too big
    if df.shape[0] > config.MAX_ROWS_POLARS_GROUPBY:
        rows_part = config.OPTIM_ROWS_POLARS_GROUPBY
        n_parts = math.ceil(df.shape[0] / rows_part)
        max_rows_part = int(config.MAX_ROWS_POLARS_GROUPBY / df.shape[0] * rows_part)
        rows_part = math.ceil(df.shape[0] / n_parts)

        log.debug(
            f'Data frame has {df.shape[0]:,} rows, more than {config.MAX_ROWS_POLARS_GROUPBY:,} '
            f'- maximum supported for aggregation. It needs to be sliced to {n_parts} parts of '
            f'{rows_part:,} and aggregated by parts. Each part then is to be truncated to max {max_rows_part:,} '
            f'rows, so that the final table has less than {config.MAX_ROWS_POLARS_GROUPBY:,}, because it will be '
            f'aggregated again.'
        )

        list_df_parts = []

        for i in tqdm(range(n_parts), desc='slice> agg > trunc', total=n_parts, unit='part'):
            df_part = df\
                .slice(i * rows_part, rows_part)\
                .groupby(['aid', 'aid_next'])\
                .sum() \
                .filter((pl.col('count') >= config.MIN_COUNT_IN_PART.get(name, 1))) \
                .sort(['count'], reverse=True)\
                .head(max_rows_part)
            list_df_parts.append(df_part)

        del df
        df = pl.concat(list_df_parts)
        del list_df_parts
        log.debug(f'{df.shape[0]:,} rows after concatenation of parts')

    df = df.groupby(['aid', 'aid_next']).sum()
    log.debug(f'{df.shape[0]:,} rows after aggregation')

    df = df\
        .filter((pl.col('count') >= config.MIN_COUNT_TO_SAVE.get(name, 1))) \
        .sort(['count'], reverse=True) \
        .head(config.MAX_CO_EVENT_PAIRS_TO_SAVE_DISK) \
        .with_column(pl.col('count').cast(pl.Int32))
    log.debug(f'{df.shape[0]:,} rows after filtering and chopping to first '
              f'{config.MAX_CO_EVENT_PAIRS_TO_SAVE_DISK:,} rows with most counts')

    df.to_pandas().to_parquet(f'{dir_stats}/{name}.parquet')  # save as panda so PyCharm can glimpse into it

    log.debug(f'df saved to {dir_stats}/{name}.parquet')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_split_alias', default='train-test')
    parser.add_argument('--count', default=True)
    parser.add_argument('--merge', default=True)
    parser.add_argument('--merge_train_test', default=True)
    args = parser.parse_args()

    dir_sessions_train = f'{config.DIR_DATA}/{args.data_split_alias}-parquet/train_sessions'
    dir_sessions_test = f'{config.DIR_DATA}/{args.data_split_alias}-parquet/test_sessions'
    dir_stats = f'{config.DIR_DATA}/{args.data_split_alias}-counts-co-event'
    folder_sessions_train = Path(dir_sessions_train).stem
    folder_sessions_test = Path(dir_sessions_test).stem
    tic_start = time.time()

    if args.count:
        log.info('count co-events per parquet file (parts) - ETA 20min')
        tic = time.time()
        count_co_events_all_files(dir_sessions_train, f'{dir_stats}/{folder_sessions_train}')
        count_co_events_all_files(dir_sessions_test, f'{dir_stats}/{folder_sessions_test}')
        log.info(f'count - time elapsed: '
                 f'{time.strftime("%Hh %Mmin %Ssec", time.gmtime(time.time() - tic))}')

    if args.merge:
        log.info('merge parts with counts and aggregate - ETA 30min')
        tic = time.time()
        for name in config.CO_EVENTS_TO_COUNT:
            concat_files_w_stats(name, f'{dir_stats}/{folder_sessions_train}')
            concat_files_w_stats(name, f'{dir_stats}/{folder_sessions_test}')
        log.info(f'merge - time elapsed: '
                 f'{time.strftime("%Hh %Mmin %Ssec", time.gmtime(time.time() - tic))}')

    if args.merge_train_test:
        log.info('merge train and test counts and aggregate')
        for name in config.CO_EVENTS_TO_COUNT:
            concat_files_w_stats(
                name=name,
                files_stats=[f'{dir_stats}/{folder_sessions_train}/{name}.parquet',
                             f'{dir_stats}/{folder_sessions_test}/{name}.parquet'],
                dir_stats=dir_stats,
            )

    log.info(f'count_co_events.py - total time elapsed: '
             f'{time.strftime("%Hh %Mmin %Ssec", time.gmtime(time.time() - tic_start))}')




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