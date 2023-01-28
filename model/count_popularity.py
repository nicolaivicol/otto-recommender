import polars as pl
import os
import glob
import logging
import argparse
import json

import config


log = logging.getLogger(os.path.basename(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_split_alias', default='train-test')
    parser.add_argument('--keep_top_n', default=20)
    args = parser.parse_args()

    keep_top_n = args.keep_top_n

    log.info(f'Running {os.path.basename(__file__)} with parameters: \n' + json.dumps(vars(args), indent=2))
    log.info('This counts popularity ranks of aids within session clusters.')

    dir_sessions = f'{config.DIR_DATA}/{args.data_split_alias}-parquet'
    dir_sessions_clusters = f'{config.DIR_DATA}/{args.data_split_alias}-sessions-clusters'
    dir_out = f'{config.DIR_DATA}/{args.data_split_alias}-counts-popularity'
    os.makedirs(dir_out, exist_ok=True)

    log.info('Load sessions with aids')
    files_sessions = sorted(glob.glob(f'{dir_sessions}/train_sessions/*.parquet')
                            + glob.glob(f'{dir_sessions}/test_sessions/*.parquet'))
    df_sessions_parts = [pl.scan_parquet(file_sessions) for file_sessions in files_sessions]
    df_sessions: pl.DataFrame = pl.concat(df_sessions_parts).collect()
    log.debug(f'Loaded df with {df_sessions.shape[0]:,} rows and {df_sessions.shape[1]} columns')

    log.info('Join sessions clusters with various n_clusters (these were precomputed with kmeans.py)')

    if 1 in config.N_CLUSTERS_TO_JOIN:
        # add a miscellaneous clusterization with one cluster only, having all sessions in it (for general popularity)
        df_sessions = df_sessions.with_column(pl.lit(0).cast(pl.Int8).alias('cl1'))

    for n_clusters in config.N_CLUSTERS_TO_JOIN:
        df_clusters = pl.read_parquet(f'{dir_sessions_clusters}/sessions-clusters-{n_clusters}.parquet')
        df_clusters = df_clusters.rename({'cluster': f'cl{n_clusters}'})
        df_sessions = df_sessions.join(df_clusters, on='session', how='left')

    df_sessions = df_sessions.fill_null(-1)

    log.info('Join ranks of AIDs by counts inside clusters...')
    time_max = df_sessions['ts'].max()
    ts_7d = time_max - 7 * 24 * 60 * 60

    for n_clusters in config.N_CLUSTERS_TO_JOIN:
        # n_clusters = 1
        log.debug(f'Join ranks within clusters out of n_clusters={n_clusters}')

        # count events by aid and clusters
        df_agg = df_sessions \
            .groupby([f'cl{n_clusters}', 'aid']) \
            .agg([
                (pl.col('type') == 0).sum().cast(pl.Int32).alias('n_clicks'),
                (pl.col('type') == 1).sum().cast(pl.Int32).alias('n_carts'),
                (pl.col('type') == 2).sum().cast(pl.Int32).alias('n_orders'),
                ((pl.col('type') == 0) & (pl.col('ts') > ts_7d)).sum().cast(pl.Int32).alias('n_clicks_7d'),
                ((pl.col('type') == 1) & (pl.col('ts') > ts_7d)).sum().cast(pl.Int32).alias('n_carts_7d'),
                ((pl.col('type') == 2) & (pl.col('ts') > ts_7d)).sum().cast(pl.Int32).alias('n_orders_7d'),
            ])

        # compute ranks of aid within clusters
        with_columns = [(pl.col(col).rank('ordinal', reverse=True).over(f'cl{n_clusters}').clip_max(999)
                         .cast(pl.Int16).alias(col.replace('n_', 'rank_')))
                        for col in df_agg.columns if col.startswith('n_')]
        df_agg = df_agg.with_columns(with_columns)
        df_agg = df_agg.rename({col: f'{col}_cl{n_clusters}' for col in df_agg.columns if col.startswith('rank_')})
        cols_rank = [col for col in df_agg.columns if col.startswith('rank_')]
        df_agg = df_agg.select(['aid', f'cl{n_clusters}'] + cols_rank)

        # keep only top N by each type and horizon
        df_agg = df_agg.filter(pl.min(cols_rank) <= keep_top_n)

        log.debug(f'Save clusters with top {keep_top_n} aids by each type/horizon: \n' + str(df_agg))
        df_agg.write_parquet(f'{dir_out}/aid_clusters_{n_clusters}_count_ranks.parquet')

    cols_cl = [col for col in df_sessions.columns if col.startswith('cl')]
    df_ses_cls = df_sessions.select(['session'] + cols_cl).unique().sort('session')
    df_ses_cls.write_parquet(f'{dir_out}/sessions_clusters.parquet')
    log.debug('Save sessions-clusters: \n' + str(df_ses_cls))
