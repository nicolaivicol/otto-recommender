import os.path
import glob
import polars as pl
from typing import List, Dict, Union
import logging
import json
from tqdm import tqdm
import argparse

import config
from feateng.w2vec import retrieve_w2vec_knns_via_faiss_index
from utils import describe_numeric

log = logging.getLogger('retrieve.py')


def get_df_count_for_co_event_type(count_type: str, dir_counts: str,  first_n: int = None) -> pl.DataFrame:
    """
    Get all pairs from a data frame with co-event counts of a given type
    :param count_type: count type, e.g. 'buy_to_buy', 'click_to_click'
    :param first_n: take first N pairs to retrieve
    :param dir_counts: dir with parquet files that contain co-event counts
    :return: pl.DataFrame(['aid', 'aid_next', 'count', 'count_pop', 'perc', 'rank', 'count_rel'])
    """
    if first_n is None:
        first_n = config.RETRIEVAL_FIRST_N_CO_COUNTS[count_type]

    df_count = pl.read_parquet(f'{dir_counts}/{count_type}.parquet')

    # over entire population
    df_count = df_count \
        .with_column((((pl.col('count') - pl.min('count')) / (pl.quantile('count', 0.9999) - pl.min('count')))
                      .clip_max(1) * 10_000).cast(pl.Int16).alias(f'{count_type}_count_pop')) \
        .with_row_count(offset=1) \
        .with_column((pl.col('row_nr') / pl.count() * 10_000).cast(pl.Int16).alias(f'{count_type}_perc_pop')) \
        .drop(f'row_nr')

    # over pairs
    df_count = df_count \
        .sort(['aid']) \
        .select([pl.all(),  # select all column from the original df
                 pl.col('count').rank('ordinal', reverse=True).over('aid').cast(pl.Int16).alias(f'{count_type}_rank'),
                 pl.col('count').max().over('aid').alias('max_count'),
                 ]) \
        .filter(pl.col(f'{count_type}_rank') <= first_n) \
        .with_columns(
            [(pl.col('count') / pl.col('max_count') * 100).cast(pl.Int8).alias(f'{count_type}_count_rel')]) \
        .drop(['max_count']) \
        .rename({'count': f'{count_type}_count'})

    return df_count


def get_pairs_for_all_co_event_types(dir_counts) -> Dict[str, pl.DataFrame]:
    """
    Get all available pairs from data frames with co-event counts for all types
    + features derived from co-counts
    """
    return {type_count: get_df_count_for_co_event_type(type_count, dir_counts)
            for type_count in config.RETRIEVAL_CO_COUNTS_TO_JOIN}


def get_pairs_co_event_type(df_aids: pl.DataFrame, df_count: pl.DataFrame, type: int) -> pl.DataFrame:
    """
    Get pairs for given AIDs and type from counts
    :param df_aids: data frame with aid and type
    :param df_count: data frame with counts of co-events
    :param type: type of first event, e.g. 0: clicks, 1: carts, 2: orders (buy)
    :return:
    """
    df_pairs = df_aids[['aid', 'type']] \
        .unique() \
        .filter(pl.col('type') == type) \
        .join(df_count[['aid', 'aid_next']], on='aid', how='inner') \
        .drop(['type'])
    return df_pairs


def get_pairs_for_all_co_event_types_for_aids(
            df_aids: pl.DataFrame,
            pairs_co_events: Dict[str, pl.DataFrame]
    ) -> pl.DataFrame:
        """
        Get pairs given AIDs from data frames with all types of co-event counts
        :param df_aids: data frame with aid and type
        :param pairs_co_events: data frames with all types of co-event counts
        :return:
        """
        lst = []
        for count_type in config.RETRIEVAL_CO_COUNTS_TO_JOIN:
            df_pairs = get_pairs_co_event_type(df_aids, pairs_co_events[count_type], config.MAP_NAME_COUNT_TYPE[count_type][0])
            lst.append(df_pairs)

        df_pairs_co_count = pl.concat(lst).unique()  # all pairs from all counted co-events
        return df_pairs_co_count


def compute_session_stats(df_test: pl.DataFrame):
    """
    Compute session stats, have as many rows as sessions (session is key)
    :param df_test: data frame with sessions (in table format, from parquet file)
    :return:
    """
    df_session = df_test \
        .groupby(['session']) \
        .agg([pl.count('aid').cast(pl.Int16).alias('n_events_session'),
              pl.n_unique('aid').cast(pl.Int16).alias('n_aids_session'),
              (pl.col('type') == 0).sum().cast(pl.Int16).alias('n_clicks_session'),
              (pl.col('type') == 1).sum().cast(pl.Int16).alias('n_carts_session'),
              (pl.col('type') == 2).sum().cast(pl.Int16).alias('n_orders_session'),
              pl.min('ts').alias('min_ts_session'),
              pl.max('ts').alias('max_ts_session'),
              ]) \
        .with_columns([(pl.col('max_ts_session') - pl.col('min_ts_session')).alias('duration_session')])
    return df_session


def keep_last_n_aids(df_sessions_aids_full: pl.DataFrame) -> pl.DataFrame:
    """
    Get last N AIDs per session (session-aid is a composite key, session-aid pairs are unique)
    :param df_sessions_aids_full: data frame with sessions (in table format, from parquet file)
    :return:
    """
    df_sessions_aids = df_sessions_aids_full \
        .groupby(['session', 'aid', 'type']) \
        .agg([pl.count('ts').cast(pl.Int16).alias('n_aid'),
              pl.max('ts').cast(pl.Int32).alias('max_ts_aid'), ]) \
        .sort(['session']) \
        .select([pl.all(),  # select all column from the original df
                 (pl.col('max_ts_aid').rank('ordinal', reverse=True).over('session').
                  clip_max(127).cast(pl.Int8).alias('ts_order_aid')), # start counting from last
                 ])

    # keep only the last N events in session (to truncate long sessions)
    df_sessions_aids = df_sessions_aids.filter(
        ((pl.col('type') == 0) & (pl.col('ts_order_aid') <= config.RETRIEVE_N_LAST_CLICKS)) |
        ((pl.col('type') == 1) & (pl.col('ts_order_aid') <= config.RETRIEVE_N_LAST_CARTS)) |
        ((pl.col('type') == 2) & (pl.col('ts_order_aid') <= config.RETRIEVE_N_LAST_ORDERS))
    )

    return df_sessions_aids


def make_unique_session_aid_pairs(df_sessions_aids: pl.DataFrame) -> pl.DataFrame:
    df_sessions_aids = df_sessions_aids \
        .groupby(['session', 'aid']) \
        .agg([
        pl.col('n_aid').sum().alias('n_aid'),
        ((pl.col('type') == 0) * pl.col('n_aid')).sum().cast(pl.Int16).alias('n_aid_clicks'),
        ((pl.col('type') == 1) * pl.col('n_aid')).sum().cast(pl.Int16).alias('n_aid_carts'),
        ((pl.col('type') == 2) * pl.col('n_aid')).sum().cast(pl.Int16).alias('n_aid_orders'),
        pl.col('max_ts_aid').max().alias('max_ts_aid'),
        pl.when(pl.col('type') == 0).then(pl.col('max_ts_aid')).otherwise(pl.lit(None)).max().alias('max_ts_aid_clicks'),
        pl.when(pl.col('type') == 1).then(pl.col('max_ts_aid')).otherwise(pl.lit(None)).max().alias('max_ts_aid_carts'),
        pl.when(pl.col('type') == 2).then(pl.col('max_ts_aid')).otherwise(pl.lit(None)).max().alias('max_ts_aid_orders'),
        ])
    return df_sessions_aids


def get_all_aid_pairs(
        df_sessions_aids: pl.DataFrame,
        pairs_co_events: Dict[str, pl.DataFrame],
        df_knns_w2vec_all: pl.DataFrame,
        df_knns_w2vec_1_2: pl.DataFrame,
    ) -> pl.DataFrame:
    """
    Create pairs
    :param df_sessions_aids: sessions with AIDs
    :param pairs_co_events: data frames with all types of co-event counts
    :param df_knns_w2vec_all: k nearest neighbours based on word2vec using all events
    :param df_knns_w2vec_1_2: k nearest neighbours based on word2vec using 1:carts and 2:orders events only
    :return:
    """
    # Pair aid with itself (aid_next=aid)
    df_pairs_self = df_sessions_aids.select(['aid']).with_columns(pl.col('aid').alias('aid_next')).unique()
    # ... this by itself provides ~3 unique candidate AIDs per session
    #  mean   std   min    5%   10%   25%   50%    95%    98%    99%    max
    # 3.250 4.420 1.000 1.000 1.000 1.000 2.000 12.000 20.000 25.000 30.000

    # Pairs from co events (aid_next based on co-counts with aid)
    df_pairs_from_co_events = get_pairs_for_all_co_event_types_for_aids(df_sessions_aids, pairs_co_events)
    # ... this by itself provides ~39 unique candidate AIDs per session
    #   mean    std   min    5%   10%    25%    50%     95%     98%     99%     max
    # 39.454 49.779 1.000 1.000 5.000 14.000 24.000 136.000 213.000 267.000 627.000

    # pairs from word2vec (aid_next based on similarity with aid)
    df_pairs_knns_w2vec_all = df_knns_w2vec_all.select(['aid', 'aid_next'])
    df_pairs_knns_w2vec_1_2 = df_knns_w2vec_1_2.select(['aid', 'aid_next'])

    # concatenate pairs from all sources
    df_pairs = pl.concat([
        df_pairs_self,
        df_pairs_from_co_events,
        df_pairs_knns_w2vec_all,
        df_pairs_knns_w2vec_1_2,
    ]).unique()

    # All sources together retrieve ~72 unique candidate AIDs per session
    #   mean    std   min    5%    10%    25%    50%     95%     98%     99%     max
    # 72.171 88.019 1.000 9.000 23.000 30.000 39.000 236.000 382.000 491.000 957.000

    return df_pairs


def keep_sessions_aids_next(df: pl.DataFrame) -> pl.DataFrame:
    """
    Keep session-aid_next only as key, remove aid, aggregate other columns
    :param df: df with sessions-aid-aid_next as key
    :return:
    """

    # Some aid_next (~5%) may appear a few times per session (paired by multiple aid)
    #  mean   std   min    5%   10%   25%   50%   95%   98%   99%    max
    # 1.197 0.680 1.000 1.000 1.000 1.000 1.000 2.000 3.000 4.000 26.000
    # df.groupby(['session', 'aid_next']).agg([pl.count().alias('count')]).sort(['count'],reverse=True).to_pandas()

    cols_after_remove_aid = list(df.columns)
    cols_after_remove_aid.remove('aid')
    # df0 = df.filter((pl.col('session') == 11117700) & (pl.col('aid_next') == 1460571)).to_pandas()

    aggs_events_session = [
        pl.count().cast(pl.Int16).alias('n_uniq_aid'),
        (pl.col('n_aid_clicks') > 0).sum().cast(pl.Int16).alias('n_uniq_aid_clicks'),
        (pl.col('n_aid_carts') > 0).sum().cast(pl.Int16).alias('n_uniq_aid_carts'),
        (pl.col('n_aid_orders') > 0).sum().cast(pl.Int16).alias('n_uniq_aid_orders'),
        (pl.col('aid') == pl.col('aid_next')).sum().cast(pl.Int8).alias('n_aid_next_is_aid'),

        pl.sum('n_aid').cast(pl.Int16).alias('n_aid'),
        pl.sum('n_aid_clicks').cast(pl.Int16).alias('n_aid_clicks'),
        pl.sum('n_aid_carts').cast(pl.Int16).alias('n_aid_carts'),
        pl.sum('n_aid_orders').cast(pl.Int16).alias('n_aid_orders'),

        pl.max('max_ts_aid').cast(pl.Int32).alias('max_ts_aid'),
        pl.max('max_ts_aid_clicks').cast(pl.Int32).alias('max_ts_aid_clicks'),
        pl.max('max_ts_aid_carts').cast(pl.Int32).alias('max_ts_aid_carts'),
        pl.max('max_ts_aid_orders').cast(pl.Int32).alias('max_ts_aid_orders'),

        pl.mean('max_ts_aid').cast(pl.Int32).alias('mean_max_ts_aid'),
        pl.mean('max_ts_aid_orders').cast(pl.Int32).alias('mean_max_ts_aid_orders'),
    ]

    # counts of co-events
    aggs_counts_co_events = []
    for count_type in config.RETRIEVAL_CO_COUNTS_TO_JOIN:
        # add counts
        aggs_counts_co_events.append(pl.sum(f'{count_type}_count').cast(pl.Int32).alias(f'{count_type}_count'))
        # add features weighted by counts
        for feat in ['count_pop', 'perc_pop', 'rank', 'count_rel']:
            aggs_counts_co_events.extend([
                ((pl.col(f'{count_type}_{feat}') * pl.col(f'{count_type}_count')).sum() /
                 pl.col(f'{count_type}_count').sum()).cast(pl.Int16).alias(f'{count_type}_{feat}'),
            ])

    # w2vec
    aggs_w2vec = [
        (pl.col('rank_w2vec_all') > 0).sum().cast(pl.Int16).alias('n_w2vec_all'),
        pl.mean('dist_w2vec_all').cast(pl.Int32).alias('dist_w2vec_all'),
        pl.mean('rank_w2vec_all').cast(pl.Int16).alias('rank_w2vec_all'),
        pl.min('rank_w2vec_all').cast(pl.Int16).alias('best_rank_w2vec_all'),

        (pl.col('rank_w2vec_1_2') > 0).sum().cast(pl.Int16).alias('n_w2vec_1_2'),
        pl.mean('dist_w2vec_1_2').cast(pl.Int32).alias('dist_w2vec_1_2'),
        pl.mean('rank_w2vec_1_2').cast(pl.Int16).alias('rank_w2vec_1_2'),
        pl.min('rank_w2vec_1_2').cast(pl.Int16).alias('best_rank_w2vec_1_2'),
    ]

    df = df.groupby(['session', 'aid_next']).agg(aggs_events_session + aggs_counts_co_events + aggs_w2vec)

    # df1 = df.filter((pl.col('session') == 11117700) & (pl.col('aid_next') == 1460571)).to_pandas()
    assert all([c in df.columns for c in cols_after_remove_aid])
    return df


def compute_recall_after_retrieval(df: pl.DataFrame, k: int = 20) -> Dict:
    """
    Compute recall without k by taking all candidates into account
    to assess the maximum achievable recall assuming a perfect ranking
    """
    r = dict()

    for type in config.TYPES:
        # type = 'carts'
        pred_exists = f'pred_{type}' in df.columns
        if not pred_exists:
            df = df.with_column((~(df['n_aid'].is_null() | (df['n_aid'] < 0))).cast(pl.Int8).alias(f'pred_{type}'))

        df_tmp = df.groupby('session').agg([
            (pl.col(f'pred_{type}')*pl.col(f'target_{type}')).sum().clip_max(k).alias('TP'),
            pl.col(f'target_{type}').sum().clip_max(k).alias('T')
        ])

        if not pred_exists:
            df = df.drop(f'pred_{type}')

        r[f'recall_{type}'] = round(df_tmp['TP'].sum() / df_tmp['T'].sum(), 5)

    r['recall'] = round(0.1 * r['recall_clicks'] + 0.3 * r['recall_carts'] + 0.6 * r['recall_orders'], 5)

    # number of AIDs in the target tables by type:
    #          mean   std   min    5%   10%   25%   50%   95%   98%    99%    max
    # clicks: 1.000 0.000 1.000 1.000 1.000 1.000 1.000 1.000 1.000  1.000  1.000
    # carts:  1.910 2.119 1.000 1.000 1.000 1.000 1.000 5.000 8.000 10.000 61.000
    # orders: 2.012 2.052 1.000 1.000 1.000 1.000 1.000 6.000 8.000 10.000 32.000

    # recall after retrieval:
    # ver 1: 2023-01-11 21:56, commit 57346077
    # {'recall_clicks': 0.49162, 'recall_carts': 0.431, 'recall_orders': 0.64224, 'recall': 0.56381}
    # {'recall_clicks': 0.49785, 'recall_carts': 0.44199, 'recall_orders': 0.65814, 'recall': 0.57727}
    # {'recall_clicks': 0.48427, 'recall_carts': 0.42086, 'recall_orders': 0.64831, 'recall': 0.56367}
    # {'recall_clicks': 0.48555, 'recall_carts': 0.43733, 'recall_orders': 0.65651, 'recall': 0.57366}

    # ver 2: 2023-01-14 01:44, commit 5e3f7030
    # {'recall_clicks': 0.52446, 'recall_carts': 0.4658, 'recall_orders': 0.67116, 'recall': 0.59488}

    # ver 3: 2023-01-14 02:24, commit 7faa73d1
    # {'recall_clicks': 0.53438, 'recall_carts': 0.47283, 'recall_orders': 0.66918, 'recall': 0.59679}
    # {'recall_clicks': 0.54341, 'recall_carts': 0.48132, 'recall_orders': 0.67977, 'recall': 0.6066}
    # {'recall_clicks': 0.53096, 'recall_carts': 0.46004, 'recall_orders': 0.6705, 'recall': 0.59341}
    # {'recall_clicks': 0.53178, 'recall_carts': 0.47703, 'recall_orders': 0.67849, 'recall': 0.60338}

    # ver 4: 2023-01-17 01:37
    # {"recall_clicks": 0.5454, "recall_carts": 0.48396, "recall_orders": 0.6868, "recall": 0.61181}

    # kaggle:
    # https://www.kaggle.com/competitions/otto-recommender-system/discussion/370116
    # for 200 candidates
    # clicks recall = 0.58486 carts recall = 0.49270 orders recall = 0.69467
    # clicks recall = 0.6506179886006195 carts recall = 0.527631391786734 orders recall = 0.7216145392798664

    return r


def retrieve_and_gen_feats(file_sessions, file_labels, file_out, aid_pairs_co_events, df_knns_w2vec_all, df_knns_w2vec_1_2):

    labels_exists = os.path.exists(file_labels)
    if labels_exists:
        df_labels = pl.read_parquet(file_labels)

    df_sessions_aids_full = pl.read_parquet(file_sessions)
    df_sessions = compute_session_stats(df_sessions_aids_full)
    df_sessions_aids = keep_last_n_aids(df_sessions_aids_full)
    df_aid_pairs = get_all_aid_pairs(df_sessions_aids, aid_pairs_co_events, df_knns_w2vec_all, df_knns_w2vec_1_2)
    df_sessions_aids = make_unique_session_aid_pairs(df_sessions_aids)

    df = df_sessions_aids.join(df_aid_pairs, on='aid', how='left')  # join pairs from co-events, etc.

    # join co-events counts as features
    cols_before_join_co_counts = list(df.columns)
    for type_count, df_count in aid_pairs_co_events.items():
        df = df.join(df_count, on=['aid', 'aid_next'], how='left')

    # join word2vec rank/distance by aid-aid_next
    df = df.join(df_knns_w2vec_all, on=['aid', 'aid_next'], how='left')
    df = df.join(df_knns_w2vec_1_2, on=['aid', 'aid_next'], how='left')

    # keep session-aid_next only as key, remove aid, aggregate other columns
    df = keep_sessions_aids_next(df)

    # join sessions stats by 'sessions' as features
    df = df.join(df_sessions, on='session', how='left')

    # add some features based on aid time
    df = df.with_columns([
        (pl.col('max_ts_session') - pl.col('max_ts_aid')).alias('since_ts_aid'),
        (pl.col('max_ts_session') - pl.col('max_ts_aid_clicks')).alias('since_ts_aid_clicks'),
        (pl.col('max_ts_session') - pl.col('max_ts_aid_carts')).alias('since_ts_aid_carts'),
        (pl.col('max_ts_session') - pl.col('max_ts_aid_orders')).alias('since_ts_aid_orders'),

        (pl.col('max_ts_aid') - pl.col('min_ts_session')).alias('since_session_start_ts_aid'),
        (pl.col('max_ts_aid_orders') - pl.col('min_ts_session')).alias('since_session_start_ts_aid_orders'),

        ((pl.col('max_ts_aid') - pl.col('min_ts_session'))
         / (pl.col('max_ts_session') - pl.col('min_ts_session') + 1) * 100
         ).cast(pl.Int8).alias('rel_pos_max_ts_aid_in_session'),

        ((pl.col('mean_max_ts_aid') - pl.col('min_ts_session'))
         / (pl.col('max_ts_session') - pl.col('min_ts_session') + 1) * 100
         ).cast(pl.Int8).alias('rel_pos_mean_max_ts_aid_in_session'),

        ((pl.col('mean_max_ts_aid_orders') - pl.col('min_ts_session'))
         / (pl.col('max_ts_session') - pl.col('min_ts_session') + 1) * 100
         ).cast(pl.Int8).alias('rel_pos_mean_max_ts_aid_orders_in_session'),
    ])
    df = df.drop(['min_ts_session', 'max_ts_session', 'max_ts_aid', 'max_ts_aid_clicks', 'max_ts_aid_carts',
                  'max_ts_aid_orders', 'mean_max_ts_aid', 'mean_max_ts_aid_orders'])

    # add info about sources of candidates
    df = df.with_columns([
        pl.lit(1).cast(pl.Int8).alias('src_any'),
        (pl.col('n_aid_next_is_aid') > 0).cast(pl.Int8).alias('src_self'),
        (pl.col('n_aid_clicks') * pl.col('click_to_click_count') > 0).cast(pl.Int8).alias('src_click_to_click'),
        (pl.col('n_aid_clicks') * pl.col('click_to_cart_or_buy_count') > 0).cast(pl.Int8).alias('src_click_to_cart_or_buy'),
        (pl.col('n_aid_carts') * pl.col('cart_to_cart_count') > 0).cast(pl.Int8).alias('src_cart_to_cart'),
        (pl.col('n_aid_carts') * pl.col('cart_to_buy_count') > 0).cast(pl.Int8).alias('src_cart_to_buy'),
        (pl.col('n_aid_orders') * pl.col('buy_to_buy_count') > 0).cast(pl.Int8).alias('src_buy_to_buy'),
        (pl.col('n_w2vec_all') > 0).cast(pl.Int8).alias('src_w2vec_all'),
        (pl.col('n_w2vec_1_2') > 0).cast(pl.Int8).alias('src_w2vec_1_2'),
    ])
    df = df.drop(['n_aid_next_is_aid'])
    df = df.with_column(pl.col([col for col in df.columns if 'src_' in col]).fill_null(pl.lit(0)))

    # add more features
    # action_num_reverse_chrono 0.658588344124041
    # aid 0.11817917885568481
    # relative_position_in_session 0.04306810550235335
    # type_weighted_log_recency_score 0.0403055170220625
    # log_recency_score 0.0028580939171428945

    # replace NULLs with -1
    df = df.fill_null(-1)
    # df = df.with_column(pl.col(df.columns).fill_null(pl.lit(-1)))

    # TODO: add new 'next_aid' per session, based on general and doc2vec cluster popularity
    # ...

    # TODO: join general and doc2vec cluster popularity features by 'next_aid'
    # ...

    # join labels for learning
    if labels_exists:
        for type, type_id in config.TYPE2ID.items():
            df_labels_type = df_labels. \
                filter(pl.col('type') == type_id). \
                with_columns([pl.lit(1).cast(pl.Int8).alias(f'target_{type}')]) \
                .drop('type')
            df = df.join(df_labels_type,
                         left_on=['session', 'aid_next'],
                         right_on=['session', 'aid'],
                         how='outer')

        cols_target = ['target_clicks', 'target_carts', 'target_orders']
        cols_src = [c for c in df.columns if 'src_' in c]
        df = df.with_column(pl.col(cols_target + cols_src).fill_null(0))
        df = df.fill_null(-1)

        recalls_after_retrieval = compute_recall_after_retrieval(df)
        log.debug(json.dumps(recalls_after_retrieval))

    log.info(f'Data frame created: {df.shape[0]:,} rows, {df.shape[1]} columns. Saving to: {file_out}')

    df = df.sort('session')  # important

    df.write_parquet(file_out)

    if labels_exists:
        df.filter(pl.col('src_any') == 1).write_parquet(file_out.replace('-retrieved', '-ltr'))

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_split_alias', default='train-test')
    parser.add_argument('--w2vec_model_all', default='word2vec-train-test-types-all-size-100-mincount-5-window-10')
    parser.add_argument('--w2vec_model_1_2', default='word2vec-train-test-types-1-2-size-100-mincount-5-window-10')

    args = parser.parse_args()

    log.info('Start retrieve.py with parameters: \n' + json.dumps(vars(args), indent=2))
    log.info('This retrieves candidates and generates features, ETA ~20min.')

    dir_sessions = f'{config.DIR_DATA}/{args.data_split_alias}-parquet/test_sessions'
    dir_labels = f'{config.DIR_DATA}/{args.data_split_alias}-parquet/test_labels'
    dir_out = f'{config.DIR_DATA}/{args.data_split_alias}-retrieved'
    os.makedirs(dir_out, exist_ok=True)
    os.makedirs(f'{config.DIR_DATA}/{args.data_split_alias}-ltr', exist_ok=True)

    # load pairs by co-event counts
    dir_counts = f'{config.DIR_DATA}/{args.data_split_alias}-counts-co-event'
    aid_pairs_co_events = get_pairs_for_all_co_event_types(dir_counts)

    # load neighbours by word2vec
    df_knns_w2vec_all = retrieve_w2vec_knns_via_faiss_index(args.w2vec_model_all)\
        .rename({'dist_w2vec': 'dist_w2vec_all', 'rank_w2vec': 'rank_w2vec_all'})

    df_knns_w2vec_1_2 = retrieve_w2vec_knns_via_faiss_index(args.w2vec_model_1_2)\
        .rename({'dist_w2vec': 'dist_w2vec_1_2', 'rank_w2vec': 'rank_w2vec_1_2'})

    files_sessions = sorted(glob.glob(f'{dir_sessions}/*.parquet'))

    for file_sessions in tqdm(files_sessions, total=len(files_sessions), unit='part'):
        retrieve_and_gen_feats(
            file_sessions=file_sessions,
            file_labels=f'{dir_labels}/{os.path.basename(file_sessions)}',
            file_out=f'{dir_out}/{os.path.basename(file_sessions)}',
            aid_pairs_co_events=aid_pairs_co_events,
            df_knns_w2vec_all=df_knns_w2vec_all,
            df_knns_w2vec_1_2=df_knns_w2vec_1_2,
        )



# ******************************************************************************

# df_agg = df_labels.filter(pl.col('type')==2).groupby(['session', 'type']).count().to_pandas()

# raise Exception('stop here - junk below')

# print(df.shape)
# print(df.head(5).to_pandas())
# print(df.tail(5).to_pandas())

# df_aids = df_aids\
#     .sort(['max_ts_aid'], reverse=True)\
#     .groupby(['session'])\
#     .agg([pl.col('aid').cumcount().alias('order_aid')])

# df_agg = df.groupby(['session', 'aid_next'])\
#     .agg([pl.count().alias('count')])\
#     .sort(['count'],reverse=True)\
#     .to_pandas()
#
# df_agg = df.groupby(['session']).agg([pl.n_unique('aid_next').alias('count')]).sort(['count'], reverse=True).to_pandas()
# print(df_agg.shape)

# from tabulate import tabulate
# summary_ = describe_numeric(df_agg[['count']], percentiles=[0.10, 0.25, 0.50, 0.95, 0.98, 0.99])
# print(tabulate(summary_, headers=summary_.columns, showindex=False, tablefmt='github'))
# print(describe_numeric(df_agg[['count']], percentiles=[0.05, 0.10, 0.25, 0.50, 0.95, 0.98, 0.99]))
