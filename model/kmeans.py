import pandas as pd
from tqdm import tqdm
import json
import os
import glob
import logging
import argparse

import polars as pl
import numpy as np
from sklearn.cluster import KMeans

import config
from utils import set_display_options
from dask_utils import set_up_dask_client

from model.w2vec import load_w2vec_model
import dask_ml.cluster
import dask.array

log = logging.getLogger(os.path.basename(__file__))
set_display_options()


# References:
# https://realpython.com/k-means-clustering-python/


def load_aid_embeddings(model_name: str) -> pl.DataFrame:
    w2vec_model = load_w2vec_model(model_name)
    words = w2vec_model.wv.index_to_key  # words sorted by "importance"
    embeddings = w2vec_model.wv.vectors
    word2idx = {word: i for i, word in enumerate(words)}  # map word to index
    map_word_embedding = {word: embeddings[word2idx[word]] for word in words}  # map word (aid) to its embedding
    df_embeddings = pl.concat([pl.DataFrame({'aid': words}, columns={'aid': pl.Int32}),
                               pl.DataFrame(np.array(list(map_word_embedding.values())))],
                              how='horizontal')
    df_embeddings = df_embeddings.rename({col: col.replace('column_', 'dim_') for col in df_embeddings.columns})
    return df_embeddings


def compute_sessions_embeddings(
        data_split_alias='train-test',
        model_name='word2vec-train-test-types-all-size-100-mincount-5-window-10',
):
    dir_sessions = f'{config.DIR_DATA}/{data_split_alias}-parquet'
    df_weights_by_type = pl.DataFrame({'type': [0, 1, 2], 'weight_type': [0.1, 0.3, 0.6]},
                                      columns={'type': pl.Int8, 'weight_type': pl.Float32})

    df_embeddings = load_aid_embeddings(model_name)
    cols_embedding = [col for col in df_embeddings.columns if col.startswith('dim_')]

    files_sessions = sorted(glob.glob(f'{dir_sessions}/train_sessions/*.parquet')
                            + glob.glob(f'{dir_sessions}/test_sessions/*.parquet'))

    for file_sessions in tqdm(files_sessions, unit='file'):
        # file_sessions = files_sessions[0]
        df_sessions = pl.read_parquet(file_sessions)
        df_sessions = df_sessions \
            .with_column(pl.col('ts').max().over('session').alias('max_ts')) \
            .with_column((1 - (pl.col('max_ts') - pl.col('ts')) / (60 * 60 * 24 * 3)).clip_min(0.10).alias('weight_time')) \
            .join(df_weights_by_type, on='type', how='left') \
            .with_column((pl.col('weight_time') * pl.col('weight_type')).alias('weight')) \
            .drop(['ts', 'type', 'max_ts', 'weight_time', 'weight_type'])
        df_sessions = df_sessions.join(df_embeddings, on='aid', how='left').fill_null(0)  # some aid do not have an embedding
        df_sessions = df_sessions \
            .groupby('session') \
            .agg([pl.sum('weight')] + [(pl.col(col) * pl.col('weight')).sum() for col in cols_embedding]) \
            .with_columns([pl.col(col) / pl.col('weight') for col in cols_embedding]) \
            .select(['session'] + cols_embedding)
        df_sessions = df_sessions \
            .with_column(pl.concat_list(cols_embedding).alias('embedding')) \
            .select(['session', 'embedding'])

        file_name_out = file_sessions.replace('-parquet/', '-sessions-w2vec/')
        os.makedirs(os.path.dirname(file_name_out), exist_ok=True)
        df_sessions.write_parquet(file_name_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_split_alias', default='train-test')
    parser.add_argument('--model_name', default='word2vec-train-test-types-all-size-100-mincount-5-window-10')
    parser.add_argument('--use_dask', default=False)
    args = parser.parse_args()

    log.info(f'Running {os.path.basename(__file__)} with parameters: \n' + json.dumps(vars(args), indent=2))
    log.info('This finds sessions clusters.')

    dask_client = set_up_dask_client() if args.use_dask else None

    dir_sessions_embeddings = f'{config.DIR_DATA}/{args.data_split_alias}-sessions-w2vec'
    dir_out = f'{config.DIR_DATA}/{args.data_split_alias}-sessions-clusters'
    os.makedirs(dir_out, exist_ok=True)

    if not os.path.exists(dir_sessions_embeddings):
        log.info(f'Compute sessions embeddings based on word2vec embeddings of AIDs in the session')
        compute_sessions_embeddings(args.data_split_alias, args.model_name)

    # cols_embedding = [f'dim_{i}' for i in range(100)]
    # cols_embedding = [col for col in df_embeddings.columns if col.startswith('dim_')]

    log.info('Load sessions embeddings')
    df_embeddings = pl.read_parquet(f'{dir_sessions_embeddings}/*/*.parquet')
    log.debug(f'Loaded {df_embeddings.shape[0]} rows and {df_embeddings.shape[1]} columns')

    log.info('Prepare data for KMeans')
    if args.use_dask:
        # df_embeddings['embedding'].to_numpy()
        X = dask.array.from_array(np.array(df_embeddings['embedding'].to_list()), chunks='80mb')
    else:
        X = df_embeddings['embedding'].to_list()

    # akeyless connect -t data-scientists@ -v ssh.prod-access.wewix.net -c /prod/BI/data-scientists-cert-issuer --tunnel=-L :2222:YOUR_MACHINE

    log.info('Fit KMeans')

    lst_n_clusters = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    res = []

    for n_clusters in tqdm(lst_n_clusters, unit='model'):
        # n_clusters = 1000
        if args.use_dask:
            km = dask_ml.cluster.KMeans(
                n_clusters=n_clusters,
                max_iter=300,
                random_state=42
            )
        else:
            km = KMeans(
                init='random',
                n_clusters=n_clusters,
                n_init='auto',
                max_iter=300,
                random_state=42,
            )

        km.fit(X)
        log.info(f'KMeans: n_clusters={n_clusters}, inertia={km.inertia_:.2f}, n_iter={km.n_iter_}')
        res.append({'n_clusters': n_clusters, 'inertia': km.inertia_, 'n_iter': km.n_iter_})

        # save on disk
        df_clusters = pl.DataFrame({'session': df_embeddings['session'], 'cluster': km.labels_})
        file_out = f'{dir_out}/sessions-clusters-{n_clusters}.parquet'
        df_clusters.write_parquet(file_out)

    pd.DataFrame(res).to_csv(f'{dir_out}/logs.csv', index=False)

    dask_client.close(60)

    # kmeans.inertia_  # The lowest SSE value
    # kmeans.n_iter_  # The number of iterations required to converge
    # kmeans.labels_
    # # array([1, 1, 1, 0, 0, 0], dtype=int32)
    # kmeans.predict([[0, 0], [12, 3]])
    # # array([1, 0], dtype=int32)
    # kmeans.cluster_centers_
    # # array([[10.,  2.], [ 1.,  2.]])

# describe_numeric(df_sessions[['n']].to_pandas())
# df_sessions.filter(pl.col('session') == 4).to_pandas()
