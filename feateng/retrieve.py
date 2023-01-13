import polars as pl
from typing import List, Dict, Union

import config
from feateng.w2vec import retrieve_w2vec_knns_via_faiss_index
from utils import describe_numeric


def get_df_count_for_co_event_type(
        count_type: str,
        first_n: int = None,
        dir_counts: str = '../data/train-test-counts-co-event',
) -> pl.DataFrame:
    """
    Get all pairs from a data frame with co-event counts of a given type
    :param count_type: count type, e.g. 'buy_to_buy', 'click_to_click'
    :param first_n: take first N pairs to retrieve
    :param dir_counts: dir with parquet files that contain co-event counts
    :return: pl.DataFrame(['aid', 'aid_next', 'count', 'count_rel', 'rank'])
    """
    if first_n is None:
        first_n = config.RETRIEVAL_FIRST_N_CO_COUNTS[count_type]

    df_count = pl.read_parquet(f'{dir_counts}/count_{count_type}.parquet')
    df_count = df_count \
        .rename({'count': f'count_{count_type}'}) \
        .sort(['aid']) \
        .select([pl.all(),  # select all column from the original df
                 pl.col(f'count_{count_type}').rank('ordinal', reverse=True).over('aid').alias(f'rank_{count_type}'),
                 pl.col(f'count_{count_type}').max().over('aid').alias('max_count'),
                 ]) \
        .filter(pl.col(f'rank_{count_type}') <= first_n) \
        .with_columns(
        [(pl.col(f'count_{count_type}') / pl.col('max_count') * 100).cast(pl.Int8).alias(f'count_rel_{count_type}')]) \
        .drop(['max_count'])
    return df_count


def get_pairs_for_all_co_event_types() -> Dict[str, pl.DataFrame]:
    """ Get all available pairs from data frames with co-event counts for all types """
    return {type_count: get_df_count_for_co_event_type(type_count) for type_count in config.RETRIEVAL_CO_COUNTS_TO_JOIN}


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

        df_pairs_co_count = pl.concat(lst).unique() # all pairs from all counted co-events
        return df_pairs_co_count


def compute_session_stats(df_test: pl.DataFrame):
    """
    Compute session stats, have as many rows as sessions (session is key)
    :param df_test: data frame with sessions (in table format, from parquet file)
    :return:
    """
    df_session = df_test \
        .groupby(['session']) \
        .agg([pl.count('aid').alias('n_events_session'),
              pl.n_unique('aid').alias('n_aids_session'),
              (pl.col('type') == 0).sum().alias('n_clicks_session'),
              (pl.col('type') == 1).sum().alias('n_carts_session'),
              (pl.col('type') == 2).sum().alias('n_orders_session'),
              pl.min('ts').alias('min_ts_session'),
              pl.max('ts').alias('max_ts_session'),
              ]) \
        .with_columns([(pl.col('max_ts_session') - pl.col('min_ts_session')).alias('duration_session')]) \
        .drop(['min_ts_session'])
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
        df_knns_w2vec: pl.DataFrame,
    ) -> pl.DataFrame:
    """
    Create pairs
    :param df_sessions_aids: sessions with AIDs
    :param pairs_co_events: data frames with all types of co-event counts
    :param df_knns_w2vec: data frame with k nearest neighbours based on word2vec
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
    df_pairs_knns_w2vec = df_knns_w2vec.select(['aid', 'aid_next'])

    # concatenate pairs from all sources
    df_pairs = pl.concat([
        df_pairs_self,
        df_pairs_from_co_events,
        df_pairs_knns_w2vec,
    ]).unique()

    # All sources together retrieve ~72 unique candidate AIDs per session
    #   mean    std   min    5%   10%    25%    50%     95%     98%     99%     max
    # 39.678 50.039 1.000 2.000 5.000 14.000 24.000 137.000 215.000 269.000 627.000
    # with w2vec:
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

    df = df.groupby(['session', 'aid_next']) \
        .agg([
        pl.count().alias('n_uniq_aid'),
        (pl.col('n_aid_clicks') > 0).sum().alias('n_uniq_aid_clicks'),
        (pl.col('n_aid_carts') > 0).sum().alias('n_uniq_aid_carts'),
        (pl.col('n_aid_orders') > 0).sum().alias('n_uniq_aid_orders'),

        pl.sum('n_aid').alias('n_aid'),
        pl.sum('n_aid_clicks').alias('n_aid_clicks'),
        pl.sum('n_aid_carts').alias('n_aid_carts'),
        pl.sum('n_aid_orders').alias('n_aid_orders'),

        pl.max('max_ts_aid').cast(pl.Int32).alias('max_ts_aid'),
        pl.max('max_ts_aid_clicks').alias('max_ts_aid_clicks'),
        pl.max('max_ts_aid_carts').alias('max_ts_aid_carts'),
        pl.max('max_ts_aid_orders').alias('max_ts_aid_orders'),

        pl.mean('count_click_to_click').cast(pl.Int32).alias('count_click_to_click'),
        pl.mean('rank_click_to_click').cast(pl.Int32).alias('rank_click_to_click'),
        pl.mean('count_rel_click_to_click').cast(pl.Int32).alias('count_rel_click_to_click'),

        pl.mean('count_click_to_cart_or_buy').cast(pl.Int32).alias('count_click_to_cart_or_buy'),
        pl.mean('rank_click_to_cart_or_buy').cast(pl.Int32).alias('rank_click_to_cart_or_buy'),
        pl.mean('count_rel_click_to_cart_or_buy').cast(pl.Int32).alias('count_rel_click_to_cart_or_buy'),

        pl.mean('count_cart_to_cart').cast(pl.Int32).alias('count_cart_to_cart'),
        pl.mean('rank_cart_to_cart').cast(pl.Int32).alias('rank_cart_to_cart'),
        pl.mean('count_rel_cart_to_cart').cast(pl.Int32).alias('count_rel_cart_to_cart'),

        pl.mean('count_buy_to_buy').cast(pl.Int32).alias('count_buy_to_buy'),
        pl.mean('rank_buy_to_buy').cast(pl.Int32).alias('rank_buy_to_buy'),
        pl.mean('count_rel_buy_to_buy').cast(pl.Int32).alias('count_rel_buy_to_buy'),

        (pl.col('rank_w2vec') > 0).sum().alias('n_w2vec'),
        pl.mean('dist_w2vec').cast(pl.Int32).alias('dist_w2vec'),
        pl.mean('rank_w2vec').cast(pl.Int32).alias('rank_w2vec'),
        pl.min('rank_w2vec').cast(pl.Int32).alias('best_rank_w2vec'),
    ])
    # df1 = df.filter((pl.col('session') == 11117700) & (pl.col('aid_next') == 1460571)).to_pandas()
    assert all([c in df.columns for c in cols_after_remove_aid])
    return df


def compute_recall_after_retrieval(df: pl.DataFrame, k: int = 20):
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
            df.drop(f'pred_{type}')

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

    return r


if __name__ == '__main__':
    word2vec_model_name = 'word2vec-train-test-types-all-size-100-mincount-5-window-10'
    df_knns_w2vec = retrieve_w2vec_knns_via_faiss_index(word2vec_model_name)
    aid_pairs_co_events = get_pairs_for_all_co_event_types()

    join_labels = True
    file_parquet_sessions = '../data/train-test-parquet/test_sessions/0500000_0600000.parquet'
    file_parquet_label = '../data/train-test-parquet/test_labels/0500000_0600000.parquet'

    df_sessions_aids_full = pl.read_parquet(file_parquet_sessions)
    if join_labels:
        df_labels = pl.read_parquet(file_parquet_label)

    df_sessions = compute_session_stats(df_sessions_aids_full)
    df_sessions_aids = keep_last_n_aids(df_sessions_aids_full)
    df_aid_pairs = get_all_aid_pairs(df_sessions_aids, aid_pairs_co_events, df_knns_w2vec)
    df_sessions_aids = make_unique_session_aid_pairs(df_sessions_aids)

    df = df_sessions_aids.join(df_aid_pairs, on='aid', how='left')  # join pairs from co-events, etc.

    # join co-events counts as features
    cols_before_join_co_counts = list(df.columns)
    for type_count, df_count in aid_pairs_co_events.items():
        df = df.join(df_count, on=['aid', 'aid_next'], how='left')

    # join word2vec rank/distance by aid-aid_next
    df = df.join(df_knns_w2vec, on=['aid', 'aid_next'], how='left')

    df = keep_sessions_aids_next(df)

    # join sessions stats by 'sessions' as features
    df = df.join(df_sessions, on='session', how='left')

    # add some features based on aid time
    df = df.with_columns([
        (pl.col('max_ts_session') - pl.col('max_ts_aid')).alias('since_ts_aid'),
        (pl.col('max_ts_session') - pl.col('max_ts_aid_clicks')).alias('since_ts_aid_clicks'),
        (pl.col('max_ts_session') - pl.col('max_ts_aid_carts')).alias('since_ts_aid_carts'),
        (pl.col('max_ts_session') - pl.col('max_ts_aid_orders')).alias('since_ts_aid_orders'),
    ])

    # replace NULLs with -1
    df = df.with_column(pl.col(df.columns).fill_null(pl.lit(-1)))

    # TODO: add other AIDs per session, based on general popularity
    # ...

    # TODO: join general popularity features by 'next_aid'
    # ...

    # join labels for learning
    if join_labels:
        for type, type_id in config.TYPE2ID.items():
            df_labels_type = df_labels.\
                filter(pl.col('type') == type_id).\
                with_columns([pl.lit(1).alias(f'target_{type}')])\
                .drop('type')
            df = df.join(df_labels_type,
                         left_on=['session', 'aid_next'],
                         right_on=['session', 'aid'],
                         how='outer')

        df = df.with_column(pl.col(['target_clicks', 'target_carts', 'target_orders']).fill_null(pl.lit(0)))

    recalls_after_retrieval = compute_recall_after_retrieval(df)
    print(recalls_after_retrieval)
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
