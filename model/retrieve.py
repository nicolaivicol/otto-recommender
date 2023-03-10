import os.path
import glob
import polars as pl
from typing import List, Dict, Union
import logging
import json
from tqdm import tqdm
import argparse

import config
from model.kmeans_sessions import load_aid_embeddings
from model.w2vec_aids import retrieve_w2vec_knns_via_faiss_index


log = logging.getLogger(os.path.basename(__file__))


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
    # df_count has columns: aid, aid_next, count

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

    df_count = df_count.select([
        'aid',
        'aid_next',
        f'{count_type}_count',  # count of aid-aid_next co-events
        f'{count_type}_count_pop',  # count of aid-aid_next pair, normalized to the max count of entire population, multiplied by 10,000
        f'{count_type}_perc_pop',  # rank of pair among the entire population of pairs, divided by total number of pairs, multiplied by 10,000
        f'{count_type}_rank',  # rank of next_aid among all next_aids for an aid
        f'{count_type}_count_rel',  # count of aid-aid_next, normalized to the maximum count of all next_aids for an aid
    ])

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
    # df_pairs = df_aids[['aid', 'type']] \
    #     .unique() \
    #     .filter(pl.col('type') == type) \
    #     .join(df_count[['aid', 'aid_next']], on='aid', how='inner') \
    #     .drop(['type'])

    df_pairs = df_aids[['aid']] \
        .unique() \
        .join(df_count[['aid', 'aid_next']], on='aid', how='inner')

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
        .with_columns([
            (pl.col('max_ts_session') - pl.col('min_ts_session')).alias('duration_session'),
            ((pl.col('n_clicks_session') == 0) & (pl.col('n_carts_session') == 0) & (pl.col('n_orders_session') > 0)).cast(pl.Int8).alias('only_orders_session'),
        ])
    return df_session


def get_session_aid_pairs_unique(df_sessions_aids_full: pl.DataFrame) -> pl.DataFrame:
    """
    Get unique session-aid pairs (session-aid is a composite key, session-aid pairs are unique)
    Some AIDs occur multiple times within a session, but we need to have a single entry for each session-aid pair,
    so we aggregate events by session-aid to describe the sequence in the newly created columns
    :param df_sessions_aids_full: data frame with sessions (in table format, from parquet file)
    :return:
    """
    df_sessions_aids = df_sessions_aids_full \
        .groupby(['session', 'aid', 'type']) \
        .agg([pl.count().cast(pl.Int16).alias('n_aid'),
              pl.max('ts').cast(pl.Int32).alias('max_ts'), ]) \
        .with_column((pl.col('max_ts').rank('ordinal', reverse=True).over(['session', 'type'])
                      .cast(pl.Int16).alias('ts_order'))) \
        .sort(['session', 'max_ts'], reverse=[False, True])

    df_sessions_aids = df_sessions_aids \
        .groupby(['session', 'aid']) \
        .agg([
            pl.col('n_aid').sum().alias('n_aid'),
            ((pl.col('type') == 0) * pl.col('n_aid')).sum().cast(pl.Int16).alias('n_aid_clicks'),
            ((pl.col('type') == 1) * pl.col('n_aid')).sum().cast(pl.Int16).alias('n_aid_carts'),
            ((pl.col('type') == 2) * pl.col('n_aid')).sum().cast(pl.Int16).alias('n_aid_orders'),

            pl.col('max_ts').max().alias('max_ts_aid'),
            pl.when(pl.col('type') == 0).then(pl.col('max_ts')).otherwise(pl.lit(None)).max().alias('max_ts_aid_clicks'),
            pl.when(pl.col('type') == 1).then(pl.col('max_ts')).otherwise(pl.lit(None)).max().alias('max_ts_aid_carts'),
            pl.when(pl.col('type') == 2).then(pl.col('max_ts')).otherwise(pl.lit(None)).max().alias('max_ts_aid_orders'),

            pl.when(pl.col('type') == 0).then(pl.col('ts_order')).otherwise(pl.lit(None)).min().alias('ts_order_aid_clicks'),
            pl.when(pl.col('type') == 1).then(pl.col('ts_order')).otherwise(pl.lit(None)).min().alias('ts_order_aid_carts'),
            pl.when(pl.col('type') == 2).then(pl.col('ts_order')).otherwise(pl.lit(None)).min().alias('ts_order_aid_orders'),
        ])

    df_sessions_aids = df_sessions_aids \
        .with_column((pl.col('max_ts_aid').rank('ordinal', reverse=True).over('session')
                      .cast(pl.Int16).alias('ts_order_aid'))) \
        .with_column((pl.col('ts_order_aid') / pl.max('ts_order_aid').over('session') * 100).round(0)
                     .cast(pl.Int8).alias('ts_order_aid_rel'))  \
        .with_column((pl.col('n_aid').rank('ordinal', reverse=True).over('session')
                      .cast(pl.Int16).alias('rank_by_n_aid'))) \
        .with_column((pl.col('n_aid_carts').rank('ordinal', reverse=True).over('session')
                      .cast(pl.Int16).alias('rank_by_n_aid_carts'))) \
        .with_column((pl.col('n_aid_orders').rank('ordinal', reverse=True).over('session')
                      .cast(pl.Int16).alias('rank_by_n_aid_orders'))) \
        .with_column((((pl.col('n_aid_carts') > 0) & (pl.col('n_aid_orders') == 0))
                      | (pl.col('max_ts_aid_carts') > pl.col('max_ts_aid_orders')))
                     .fill_null(0).cast(pl.Int8).alias('left_in_cart')) \
        .sort(['session', 'max_ts_aid'], reverse=True)

    df_sessions_aids = df_sessions_aids \
        .with_columns([
            pl.min('max_ts_aid').over('session').alias('min_ts_session'),
            pl.max('max_ts_aid').over('session').alias('max_ts_session'),
            ]) \
        .with_column(((pl.col('max_ts_session') - pl.col('max_ts_aid'))
                      / ((pl.col('max_ts_session') - pl.col('min_ts_session')).clip_min(60 * 60)) * 100).round(0)
                     .cast(pl.Int8, strict=False).alias('ts_aid_rel_pos_in_session')) \
        .drop(['min_ts_session', 'max_ts_session'])

    # keep only the last N aids in session (to truncate too long sessions)
    df_sessions_aids = df_sessions_aids.filter(
        ((pl.col('ts_order_aid_clicks') <= config.RETRIEVE_N_LAST_CLICKS)) |
        ((pl.col('ts_order_aid_carts') <= config.RETRIEVE_N_LAST_CARTS)) |
        ((pl.col('ts_order_aid_orders') <= config.RETRIEVE_N_LAST_ORDERS)) |
        ((pl.col('rank_by_n_aid') <= config.RETRIEVE_N_MOST_FREQUENT)) |
        ((pl.col('rank_by_n_aid_carts') <= config.RETRIEVE_N_MOST_FREQUENT)) |
        ((pl.col('rank_by_n_aid_orders') <= config.RETRIEVE_N_MOST_FREQUENT))
    )

    df_sessions_aids = df_sessions_aids \
        .select([
            'session',
            'aid',
            'n_aid',
            'n_aid_clicks',
            'n_aid_carts',
            'n_aid_orders',
            'rank_by_n_aid',
            'rank_by_n_aid_carts',
            'rank_by_n_aid_orders',
            'max_ts_aid',
            'max_ts_aid_clicks',
            'max_ts_aid_carts',
            'max_ts_aid_orders',
            'ts_aid_rel_pos_in_session',
            'ts_order_aid',
            'ts_order_aid_rel',
            'ts_order_aid_clicks',
            'ts_order_aid_carts',
            'ts_order_aid_orders',
            'left_in_cart',
        ])

    return df_sessions_aids

# # ******************************************************************************************************************
# # df_sessions_aids.describe()
# import random
# y = df_sessions_aids.filter(pl.col('max_ts_aid_carts') > pl.col('max_ts_aid_orders'))['session']
# x = random.choice(y)
# df_sessions_aids.filter(pl.col('session') == x).to_pandas()
# df_sessions_aids_full.filter(pl.col('session') == x).sort('ts', reverse=True).to_pandas()
# # ******************************************************************************************************************


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
    df_pairs_self = df_sessions_aids.select(['aid']).with_column(pl.col('aid').alias('aid_next')).unique()
    # ... this by itself provides ~3.5 unique candidate AIDs per session, see summary stats below:
    #  mean   std   min    1%    5%   10%   25%   50%   75%    95%    98%    99%     max
    # 3.458 6.038 1.000 1.000 1.000 1.000 1.000 2.000 3.000 12.000 20.000 29.000 117.000

    # Pairs from co events (aid_next based on co-counts with aid)
    df_pairs_from_co_events = get_pairs_for_all_co_event_types_for_aids(df_sessions_aids, pairs_co_events)
    #   mean     std   min    1%    5%    10%    25%    50%    75%     95%     98%     99%      max
    # 72.393 114.139 1.000 1.000 4.000 12.000 25.000 38.000 75.000 243.000 402.000 558.000 2419.000

    # Pairs from word2vec (aid_next based on similarity with aid)
    df_pairs_knns_w2vec_all = df_knns_w2vec_all.select(['aid', 'aid_next'])
    #   mean    std   min    1%    5%    10%    25%    50%    75%     95%     98%     99%      max
    # 53.207 82.146 1.000 1.000 1.000 20.000 20.000 21.000 57.000 174.000 288.000 404.000 1618.000

    df_pairs_knns_w2vec_1_2 = df_knns_w2vec_1_2.select(['aid', 'aid_next'])
    #   mean    std   min    1%    5%   10%    25%    50%    75%     95%     98%     99%      max
    # 52.105 84.718 1.000 1.000 1.000 1.000 20.000 20.000 58.000 178.000 295.000 411.000 1742.000

    # concatenate pairs from all sources
    df_pairs = pl.concat([
        df_pairs_self,
        df_pairs_from_co_events,
        df_pairs_knns_w2vec_all,
        df_pairs_knns_w2vec_1_2,
    ]).unique()

    # All sources together retrieve ~149 unique candidate AIDs per session
    #    mean     std   min    10%    25%    50%     75%     95%     98%      99%      max
    # 148.981 232.118 1.000 38.000 55.000 71.000 153.000 493.000 817.000 1141.010 4579.000

    return df_pairs


def keep_sessions_aids_next(df: pl.DataFrame) -> pl.DataFrame:
    """
    Keep session-aid_next only as key, remove aid, aggregate other columns
    :param df: df with sessions-aid-aid_next as key
    :return:
    """

    # Some aid_next (~5%) may appear a few times per session (paired by multiple aid)
    #  mean   std   min    5%   25%   50%   75%   90%   95%   98%   99%    max
    # 1.166 0.625 1.000 1.000 1.000 1.000 1.000 2.000 2.000 3.000 4.000 44.000

    cols_after_remove_aid = list(df.columns)
    cols_after_remove_aid.remove('aid')

    # when aid=aid_next, compute features about the item itself within the session
    # (how many times it occured, how recent, was left in the cart, etc.)
    feats_self = [
        (pl.col('aid') == pl.col('aid_next')).sum().cast(pl.Int8).alias('aid_next_is_aid'),
        # counts of events, total and by type
        ((pl.col('aid') == pl.col('aid_next')) * pl.col('n_aid')).sum().cast(pl.Int16).alias('slf_n'),
        ((pl.col('aid') == pl.col('aid_next')) * pl.col('n_aid_clicks')).sum().cast(pl.Int16).alias('slf_n_clicks'),
        ((pl.col('aid') == pl.col('aid_next')) * pl.col('n_aid_carts')).sum().cast(pl.Int16).alias('slf_n_carts'),
        ((pl.col('aid') == pl.col('aid_next')) * pl.col('n_aid_orders')).sum().cast(pl.Int16).alias('slf_n_orders'),
        # ranks by counts
        ((pl.col('aid') == pl.col('aid_next')) * pl.col('rank_by_n_aid')).min().cast(pl.Int16).alias('slf_rank_by_n'),
        ((pl.col('aid') == pl.col('aid_next')) * pl.col('rank_by_n_aid_carts')).min().cast(pl.Int16).alias('slf_rank_by_n_carts'),
        ((pl.col('aid') == pl.col('aid_next')) * pl.col('rank_by_n_aid_orders')).min().cast(pl.Int16).alias('slf_rank_by_n_orders'),
        # when it occured last time
        ((pl.col('aid') == pl.col('aid_next')) * pl.col('max_ts_aid')).max().cast(pl.Int32).alias('slf_max_ts'),
        ((pl.col('aid') == pl.col('aid_next')) * pl.col('max_ts_aid_clicks')).max().cast(pl.Int32).alias('slf_max_ts_clicks'),
        ((pl.col('aid') == pl.col('aid_next')) * pl.col('max_ts_aid_carts')).max().cast(pl.Int32).alias('slf_max_ts_carts'),
        ((pl.col('aid') == pl.col('aid_next')) * pl.col('max_ts_aid_orders')).max().cast(pl.Int32).alias('slf_max_ts_orders'),
        # when it occured last time, relative to the session
        ((pl.col('aid') == pl.col('aid_next')) * pl.col('ts_aid_rel_pos_in_session')).min().cast(pl.Int16).alias('slf_ts_rel_pos_in_session'),
        ((pl.col('aid') == pl.col('aid_next')) * pl.col('ts_order_aid')).min().cast(pl.Int16).alias('slf_ts_order'),
        ((pl.col('aid') == pl.col('aid_next')) * pl.col('ts_order_aid_rel')).min().cast(pl.Int16).alias('slf_ts_order_rel'),
        ((pl.col('aid') == pl.col('aid_next')) * pl.col('ts_order_aid_clicks')).min().cast(pl.Int16).alias('slf_ts_order_clicks'),
        ((pl.col('aid') == pl.col('aid_next')) * pl.col('ts_order_aid_carts')).min().cast(pl.Int16).alias('slf_ts_order_carts'),
        ((pl.col('aid') == pl.col('aid_next')) * pl.col('ts_order_aid_orders')).min().cast(pl.Int16).alias('slf_ts_order_orders'),
        # whether it was left in cart
        ((pl.col('aid') == pl.col('aid_next')) * pl.col('left_in_cart')).sum().cast(pl.Int8).alias('slf_left_in_cart'),
    ]

    # when multiple aids recommending the same aid_next, aggregate features
    aggs_events_session = [
        pl.count().cast(pl.Int16).alias('n_uniq_aid'),
        (pl.col('n_aid_clicks') > 0).sum().cast(pl.Int16).alias('n_uniq_aid_clicks'),
        (pl.col('n_aid_carts') > 0).sum().cast(pl.Int16).alias('n_uniq_aid_carts'),
        (pl.col('n_aid_orders') > 0).sum().cast(pl.Int16).alias('n_uniq_aid_orders'),

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

        pl.min('ts_order_aid').cast(pl.Int16).alias('ts_order_aid'),
        pl.min('ts_order_aid_rel').cast(pl.Int16).alias('ts_order_aid_rel'),
        pl.min('ts_order_aid_clicks').cast(pl.Int16).alias('ts_order_aid_clicks'),
        pl.min('ts_order_aid_carts').cast(pl.Int16).alias('ts_order_aid_carts'),
        pl.min('ts_order_aid_orders').cast(pl.Int16).alias('ts_order_aid_orders'),
        pl.mean('ts_aid_rel_pos_in_session').cast(pl.Int16).alias('ts_aid_rel_pos_in_session'),

        pl.min('rank_by_n_aid').cast(pl.Int16).alias('rank_by_n_aid'),
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

    df_session_aid_next = df \
        .groupby(['session', 'aid_next']) \
        .agg(feats_self + aggs_events_session + aggs_counts_co_events + aggs_w2vec)

    # missing_cols = [c for c in cols_after_remove_aid if c not in df_session_aid_next.columns]
    # assert len(missing_cols) == 0, f'columns missing from agregation: {",".join(missing_cols)}'

    # df1 = df.filter((pl.col('session') == 11117700) & (pl.col('aid_next') == 1460571)).to_pandas()
    # df2 = df_session_aid_next.filter((pl.col('session') == 11117700) & (pl.col('aid_next') == 1460571)).to_pandas()
    # from utils import describe_numeric
    # describe_numeric(df.groupby(['session', 'aid_next']).agg([pl.count().alias('count')]).sort(['count'],reverse=True).to_pandas()[['count']], percentiles=[0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.98, 0.99])

    return df_session_aid_next


def stats_number_of_aid_next_per_session(df):
    from utils import describe_numeric
    df_agg = df.groupby(['session']).agg([pl.n_unique('aid_next').alias('count')]).sort(['count'], reverse=True).to_pandas()
    summary_ = describe_numeric(df_agg[['count']], percentiles=[0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.95, 0.98, 0.99])
    return summary_


def load_sessions_embeddings(file):
    """Load session embeddings from parquet file"""
    df_session_embeddings = pl.read_parquet(file)
    size = len(df_session_embeddings[0, 'embedding'])
    cols = [pl.col('session')] + [pl.col('embedding').arr.get(i).alias(f'dim_{i}_ses') for i in range(size)]
    df_session_embeddings = df_session_embeddings.select(cols)  # list from embedding to separate columns
    return df_session_embeddings


def retrieve_and_gen_feats(
        df_sessions_aids_full: pl.DataFrame,
        aid_pairs_co_events: Dict[str, pl.DataFrame],
        df_knns_w2vec_all: pl.DataFrame,
        df_knns_w2vec_1_2: pl.DataFrame,
        df_session_cl: pl.DataFrame,
        df_pop_cl1: pl.DataFrame,
        df_pop_cl50: pl.DataFrame,
        df_aid_embeddings: pl.DataFrame,
        df_session_embeddings: pl.DataFrame,
        df_labels: pl.DataFrame,
        file_out: str,
):
    """
    This function retrieves candidates for each session and attaches features.

    The function first checks if the labels data frame exists and reads it if it does.
    It then reads the sessions data frame and gets unique session-aid pairs.
    It then gets all aid-aid_next pairs from self-join, co-events, and word2vec.
    The function then joins co-event counts and word2vec rank/distance as features.
    It trims weaker pairs so that more recent AIDs have more pairs and older AIDs have less pairs.
    The function then keeps session-aid_next only as key, removes aid, and aggregates other columns.

    Parameters
    ----------
    df_sessions_aids_full: pl.DataFrame([session, aid, ts, type])
        all sessions with items (aid) and their types
    aid_pairs_co_events
        a dictionary of data frames containing co-event counts for different event types
    df_knns_w2vec_all
        data frame containing all aid-aid_next pairs with their word2vec rank and distance, for all types
    df_knns_w2vec_1_2
        data frame containing aid-aid_next pairs that with their word2vec rank and distance, for types carts and orders
    df_session_cl
        data frame containing session-cluster information
    df_pop_cl1
        data frame containing general popularity of aid (item) by type (1 cluster, i.e. general for all sessions)
    df_pop_cl50
        data frame containing popularity of aid (item) by type, in 50 clusters of sessions
    df_aid_embeddings: pl.DataFrame([aid, dim_0_aid, dim_1_aid, ...])
        embeddings of aids (items)
    df_session_embeddings: pl.DataFrame([session, dim_0_ses, dim_1_ses, ...])
        embeddings of sessions
    df_labels: pl.DataFrame([session, aid_next, label])
        labels for the session-aid_next pairs
    file_out: str
        file path of the output data frame in parquet format

    Returns
    -------
    df:
        It saves the output data frame in the specified file path in parquet format.
    """

    df_sessions_aids = get_session_aid_pairs_unique(df_sessions_aids_full)
    df_aid_pairs = get_all_aid_pairs(df_sessions_aids, aid_pairs_co_events, df_knns_w2vec_all, df_knns_w2vec_1_2)

    # # join all found pairs from self-join, co-events, word2vec.
    df = df_sessions_aids.join(df_aid_pairs, on='aid', how='left')

    # join co-events counts as features
    for type_count, df_count in aid_pairs_co_events.items():
        df = df.join(df_count, on=['aid', 'aid_next'], how='left')

    # join word2vec rank/distance by aid-aid_next
    df = df.join(df_knns_w2vec_all, on=['aid', 'aid_next'], how='left')
    df = df.join(df_knns_w2vec_1_2, on=['aid', 'aid_next'], how='left')

    # Trim weaker pairs so that we keep more pairs for more recent AIDs and less pairs for older AIDs (worse order).
    # Scheme: 'aid' with order=1 ('best_order_aid'=1) gets at most 20 aid_next, aid with order 30 or lower gets at most 3 aid_next
    # Formula: best_co_count_rank <= max(3, 20 - (20 - 3) / (30 - 1) * best_order_aid)
    max_n_aid_next_for_order_1 = 20
    min_n_aid_next = 3
    min_n_aid_next_at_order = 20
    delta_per_order = (max_n_aid_next_for_order_1 - min_n_aid_next) / (min_n_aid_next_at_order - 1)
    cols_order = ['rank_by_n_aid', 'ts_order_aid', 'ts_order_aid_clicks', 'ts_order_aid_carts', 'ts_order_aid_orders']
    df = df.with_column(pl.min(cols_order).alias('best_order_aid'))
    df = df.with_column((max_n_aid_next_for_order_1 - delta_per_order * (pl.col('best_order_aid') - 1)).clip_min(min_n_aid_next).alias('best_order_aid_th'))

    cols_rank_count = ['click_to_click_rank', 'click_to_cart_or_buy_rank', 'cart_to_cart_rank', 'cart_to_buy_rank', 'buy_to_buy_rank']
    df = df.with_column(pl.min(cols_rank_count).alias('best_co_count_rank'))

    cols_w2vec = ['rank_w2vec_all', 'rank_w2vec_1_2']
    df = df.with_column(pl.min(cols_w2vec).alias('best_w2vec_rank'))

    df = df.filter(
        (pl.col('aid') == pl.col('aid_next'))
        | (pl.col('best_co_count_rank') <= pl.col('best_order_aid_th'))
        | (pl.col('best_w2vec_rank') <= pl.col('best_order_aid_th')))

    # stats for the number of aid_next per session after trimming (15% of pairs):
    #    mean     std   min    1%     5%    10%    25%    50%     75%     95%     98%     99%      max
    # 130.894 148.970 1.000 1.000 21.000 38.000 55.000 71.000 149.000 426.000 626.000 776.000 1763.000

    df = df.drop(['best_order_aid', 'best_order_aid_th', 'best_co_count_rank', 'best_w2vec_rank'])

    # keep session-aid_next only as key, remove aid, aggregate other columns
    df = keep_sessions_aids_next(df)

    # join sessions stats by 'sessions' as features
    df_session_stats = compute_session_stats(df_sessions_aids_full)
    df = df.join(df_session_stats, on='session', how='left')

    # add some features based on aid time
    df = df.with_columns([
        (pl.col('max_ts_session') - pl.col('max_ts_aid')).alias('since_ts_aid'),
        (pl.col('max_ts_session') - pl.col('max_ts_aid_clicks')).alias('since_ts_aid_clicks'),
        (pl.col('max_ts_session') - pl.col('max_ts_aid_carts')).alias('since_ts_aid_carts'),
        (pl.col('max_ts_session') - pl.col('max_ts_aid_orders')).alias('since_ts_aid_orders'),

        (pl.col('max_ts_session') - pl.col('slf_max_ts')).alias('slf_since_ts'),
        (pl.col('max_ts_session') - pl.col('slf_max_ts_clicks')).alias('slf_since_ts_clicks'),
        (pl.col('max_ts_session') - pl.col('slf_max_ts_carts')).alias('slf_since_ts_carts'),
        (pl.col('max_ts_session') - pl.col('slf_max_ts_orders')).alias('slf_since_ts_orders'),

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
                  'max_ts_aid_orders', 'mean_max_ts_aid', 'mean_max_ts_aid_orders',
                  'slf_max_ts', 'slf_max_ts_clicks', 'slf_max_ts_carts', 'slf_max_ts_orders',
                  ])

    # add info about sources of candidates
    df = df.with_columns([
        pl.lit(1).cast(pl.Int8).alias('src_any'),
        (pl.col('aid_next_is_aid') > 0).cast(pl.Int8).alias('src_self'),
        (pl.col('n_aid_clicks') * pl.col('click_to_click_count') > 0).cast(pl.Int8).alias('src_click_to_click'),
        (pl.col('n_aid_clicks') * pl.col('click_to_cart_or_buy_count') > 0).cast(pl.Int8).alias('src_click_to_cart_or_buy'),
        (pl.col('n_aid_carts') * pl.col('cart_to_cart_count') > 0).cast(pl.Int8).alias('src_cart_to_cart'),
        (pl.col('n_aid_carts') * pl.col('cart_to_buy_count') > 0).cast(pl.Int8).alias('src_cart_to_buy'),
        (pl.col('n_aid_orders') * pl.col('buy_to_buy_count') > 0).cast(pl.Int8).alias('src_buy_to_buy'),
        (pl.col('n_w2vec_all') > 0).cast(pl.Int8).alias('src_w2vec_all'),
        (pl.col('n_w2vec_1_2') > 0).cast(pl.Int8).alias('src_w2vec_1_2'),
    ])
    df = df.drop(['aid_next_is_aid'])

    # add new 'next_aid' per session, based on popularity within doc2vec cluster of the session
    df = df.join(df_session_cl.select(['session', 'cl50']), on='session', how='left')

    df_ses_cl50_aid = df \
        .select(['session', 'cl50']) \
        .unique() \
        .join(df_pop_cl50.rename({'aid': 'aid_next'}), on='cl50') \
        .with_column(pl.lit(1).cast(pl.Int8).alias('src_pop_cl50'))

    # keep only top 20 candidates by each type of rank
    cols_rank = [col for col in df_ses_cl50_aid.columns if (col.startswith('rank_') & col.endswith('_cl50'))]
    df_ses_cl50_aid = df_ses_cl50_aid.filter(pl.min(cols_rank) <= 20)

    # join new candidates with their ranks within clusters
    df = df.join(df_ses_cl50_aid, on=['session', 'cl50', 'aid_next'], how='outer').drop('cl50')

    # add general popularity ranks as features, without adding new candidates
    df = df.join(
        df_pop_cl1.select(['aid', 'rank_clicks_cl1', 'rank_carts_cl1', 'rank_orders_cl1']),
        left_on='aid_next', right_on='aid', how='left')

    # mark all candidates with 'src_any'=1
    df = df.with_column(pl.lit(1).cast(pl.Int8).alias('src_any'))

    # fill NULLs with 0 for src_ columns
    df = df.with_column(pl.col([col for col in df.columns if 'src_' in col]).fill_null(0))

    # fill timestamp order with 999 to have candidates from cluster last when sorting before saving
    df = df.with_column(pl.col('ts_order_aid').fill_null(999))

    # replace NULLs with -1 for other columns
    df = df.fill_null(-1)

    # join similarity between candidates and session (cosine similarity, euclidean distance)
    parts_df_ses_aid_sim = []

    for i in range(0, len(df), 100_000):
        df_ses_aid_sim = df[i:min(i + 100_000, len(df)), ['session', 'aid_next']] \
            .unique() \
            .join(df_session_embeddings, on='session', how='inner') \
            .join(df_aid_embeddings, left_on='aid_next', right_on='aid', how='inner', suffix='_aid') \
            .with_column(pl.sum([pl.col(f'dim_{i}_ses') * pl.col(f'dim_{i}_aid') for i in range(100)]).alias('dot')) \
            .with_column(pl.sum([pl.col(f'dim_{i}_ses') * pl.col(f'dim_{i}_ses') for i in range(100)]).sqrt().alias('norm_ses')) \
            .with_column(pl.sum([pl.col(f'dim_{i}_aid') * pl.col(f'dim_{i}_aid') for i in range(100)]).sqrt().alias('norm_aid')) \
            .with_column((pl.col('dot') / (pl.col('norm_ses') * pl.col('norm_aid'))).alias('cos_sim_ses_aid')) \
            .with_column(pl.sum([(pl.col(f'dim_{i}_ses') - pl.col(f'dim_{i}_aid')).pow(2) for i in range(100)]).sqrt().alias('eucl_dist_ses_aid'))\
            .select(['session', 'aid_next', 'cos_sim_ses_aid', 'eucl_dist_ses_aid'])

        parts_df_ses_aid_sim.append(df_ses_aid_sim)

    df_ses_aid_sim = pl.concat(parts_df_ses_aid_sim)

    df = df.join(df_ses_aid_sim, on=['session', 'aid_next'], how='left') \
        .with_column(pl.col('cos_sim_ses_aid').fill_null(0)) \
        .with_column(pl.col('eucl_dist_ses_aid').fill_null(-1))

    # TODO: compute ranks by main features, then compute a weighted rank out of these ranks, sort by it

    # if available, join labels for learning-to-rank
    if df_labels is not None:

        for type, type_id in config.TYPE2ID.items():
            col_target = f'target_{type}'

            df_labels_type = df_labels \
                .filter(pl.col('type') == type_id) \
                .with_columns([pl.lit(1).cast(pl.Int8).alias(col_target)]) \
                .select(['session', 'aid', col_target])

            df = df.join(df_labels_type, left_on=['session', 'aid_next'], right_on=['session', 'aid'], how='left')

        # fill NULLs for target_ columns with 0
        cols_target = [col for col in df.columns if col.startswith('target_')]
        df = df.with_column(pl.col(cols_target).fill_null(config.FILL_NULL_TARGET_WITH_VALUE))

    log.info(f'Data frame created: {df.shape[0]:,} rows, {df.shape[1]} columns.')

    log.debug('Sorting by session and ts_order_aid...')
    # important to sort by session (sorting by ts_order_aid is optional, but helps
    # to see a more meaningful recall@k when evaluating the retrieved candidates
    df = df.sort(['session', 'ts_order_aid'])

    log.info(f'Saving to: {file_out}')
    # save data ready for learning-to-rank models (e.g. lightgbm.dask.DaskLGBMRanker)
    df.write_parquet(file_out)

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_split_alias', default='train-test')
    parser.add_argument('--w2vec_model_all', default='word2vec-train-test-types-all-size-100-mincount-5-window-10')
    parser.add_argument('--w2vec_model_1_2', default='word2vec-train-test-types-1-2-size-100-mincount-5-window-10')
    # python -m model.retrieve --data_split_alias full --w2vec_model_all word2vec-full-types-all-size-100-mincount-5-window-10 --w2vec_model_1_2 word2vec-full-types-1-2-size-100-mincount-5-window-10

    args = parser.parse_args()

    log.info('Start retrieve.py with parameters: \n' + json.dumps(vars(args), indent=2))
    log.info('This retrieves candidates and generates features, ETA ~40min.')

    dir_sessions = f'{config.DIR_DATA}/{args.data_split_alias}-parquet/test_sessions'
    dir_labels = f'{config.DIR_DATA}/{args.data_split_alias}-parquet/test_labels'
    dir_sessions_embeddings = f'{config.DIR_DATA}/{args.data_split_alias}-sessions-w2vec-parquet/test_sessions'
    dir_out = f'{config.DIR_DATA}/{args.data_split_alias}-retrieved'
    os.makedirs(dir_out, exist_ok=True)

    # load pairs by co-event counts
    dir_counts = f'{config.DIR_DATA}/{args.data_split_alias}-counts-co-event'
    aid_pairs_co_events = get_pairs_for_all_co_event_types(dir_counts)

    # load neighbours by word2vec
    df_knns_w2vec_all = retrieve_w2vec_knns_via_faiss_index(args.w2vec_model_all)\
        .rename({'dist_w2vec': 'dist_w2vec_all', 'rank_w2vec': 'rank_w2vec_all'})

    df_knns_w2vec_1_2 = retrieve_w2vec_knns_via_faiss_index(args.w2vec_model_1_2)\
        .rename({'dist_w2vec': 'dist_w2vec_1_2', 'rank_w2vec': 'rank_w2vec_1_2'})

    # load aids by popularity within sessions clusters (based on word2vec embeddings of aids within session)
    dir_counts_pop = f'{config.DIR_DATA}/{args.data_split_alias}-counts-popularity'
    df_session_cl = pl.read_parquet(f'{dir_counts_pop}/sessions_clusters.parquet')
    df_pop_cl50 = pl.read_parquet(f'{dir_counts_pop}/aid_clusters_50_count_ranks.parquet')
    df_pop_cl1 = pl.read_parquet(f'{dir_counts_pop}/aid_clusters_1_count_ranks.parquet')

    # load all available embeddings of aids (items)
    df_aid_embeddings = load_aid_embeddings(args.w2vec_model_all, col_sufix='_aid')

    files_sessions = sorted(glob.glob(f'{dir_sessions}/*.parquet'))

    for file_sessions in tqdm(files_sessions, unit='part', leave=False):

        df_sessions_aids_full = pl.read_parquet(file_sessions)  # has columns: [session, aid, ts, type]
        df_session_embeddings = load_sessions_embeddings(f'{dir_sessions_embeddings}/{os.path.basename(file_sessions)}')
        file_labels = f'{dir_labels}/{os.path.basename(file_sessions)}'
        df_labels = pl.read_parquet(file_labels) if os.path.exists(file_labels) else None

        retrieve_and_gen_feats(
            df_sessions_aids_full=df_sessions_aids_full,
            aid_pairs_co_events=aid_pairs_co_events,
            df_knns_w2vec_all=df_knns_w2vec_all,
            df_knns_w2vec_1_2=df_knns_w2vec_1_2,
            df_session_cl=df_session_cl,
            df_pop_cl1=df_pop_cl1,
            df_pop_cl50=df_pop_cl50,
            df_aid_embeddings=df_aid_embeddings,
            df_session_embeddings=df_session_embeddings,
            df_labels=df_labels,
            file_out=f'{dir_out}/{os.path.basename(file_sessions)}',
        )


# ******************************************************************************
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

# print(df_agg.shape)

# from tabulate import tabulate


# print(tabulate(summary_, headers=summary_.columns, showindex=False, tablefmt='github'))
# print(describe_numeric(df_agg[['count']], percentiles=[0.05, 0.10, 0.25, 0.50, 0.95, 0.98, 0.99]))
