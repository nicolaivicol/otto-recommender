import numpy
import pandas as pd
from tqdm import tqdm
import json
import os
import glob
import logging
import argparse

import h5py
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
# https://dask.pydata.org/en/latest/array-creation.html


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
            .with_column((pl.col('weight_time') * pl.col('weight_type')).cast(pl.Float32).alias('weight')) \
            .drop(['ts', 'type', 'max_ts', 'weight_time', 'weight_type'])
        df_sessions = df_sessions.join(df_embeddings, on='aid', how='left').fill_null(0)  # some aid do not have an embedding
        df_sessions = df_sessions \
            .groupby('session') \
            .agg([pl.sum('weight')] + [(pl.col(col) * pl.col('weight')).sum() for col in cols_embedding]) \
            .with_columns([(pl.col(col) / pl.col('weight')).round(6).cast(pl.Float32) for col in cols_embedding]) \
            .select(['session'] + cols_embedding)
        df_sessions = df_sessions \
            .with_column(pl.concat_list(cols_embedding).alias('embedding')) \
            .select(['session', 'embedding'])\
            .sort('session')

        # save to .parquet file
        file_name_out_parquet = file_sessions.replace('-parquet/', '-sessions-w2vec-parquet/')
        os.makedirs(os.path.dirname(file_name_out_parquet), exist_ok=True)
        df_sessions.write_parquet(file_name_out_parquet)

        # save to hdf5 file
        np_embedding = np.array(df_sessions['embedding'].to_list())
        np_session = df_sessions['session'].to_numpy()
        file_name_out_hdf5 = file_sessions.replace('-parquet/', '-sessions-w2vec-h5/').replace('.parquet', '.h5')
        os.makedirs(os.path.dirname(file_name_out_hdf5), exist_ok=True)
        with h5py.File(file_name_out_hdf5, 'w') as hf:
            hf.create_dataset('session', data=np_session, dtype='int32')
            hf.create_dataset('embedding', data=np_embedding, dtype='float32')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_split_alias', default='train-test')
    parser.add_argument('--model_name', default='word2vec-train-test-types-all-size-100-mincount-5-window-10')
    parser.add_argument('--use_dask', default=True)
    args = parser.parse_args()

    use_dask = args.use_dask

    log.info(f'Running {os.path.basename(__file__)} with parameters: \n' + json.dumps(vars(args), indent=2))
    log.info('This finds sessions clusters.')

    dask_client = set_up_dask_client() if use_dask else None

    dir_sessions_embeddings_h5 = f'{config.DIR_DATA}/{args.data_split_alias}-sessions-w2vec-h5'
    dir_sessions_embeddings_parquet = f'{config.DIR_DATA}/{args.data_split_alias}-sessions-w2vec-parquet'
    dir_out = f'{config.DIR_DATA}/{args.data_split_alias}-sessions-clusters'
    os.makedirs(dir_out, exist_ok=True)

    files_w_embeddings_missing = (
            (use_dask and not os.path.exists(dir_sessions_embeddings_h5))
            or (not use_dask and not os.path.exists(dir_sessions_embeddings_parquet)))

    if files_w_embeddings_missing:
        log.info(f'Compute sessions embeddings based on word2vec embeddings of AIDs in the session')
        compute_sessions_embeddings(args.data_split_alias, args.model_name)

    log.info('Load sessions embeddings')

    if use_dask:
        files_h5 = sorted(glob.glob(f'{dir_sessions_embeddings_h5}/*/*.h5'))
        vecs_parts = []
        sessions_parts = []

        for f in files_h5:
            hf = h5py.File(f)
            vecs_parts.append(dask.array.from_array(hf['embedding'], chunks=(100_000, 100)))
            sessions_parts.append(dask.array.from_array(hf['session']))

        X = dask.array.concatenate(vecs_parts)
        sessions = dask.array.concatenate(sessions_parts).compute()
        log.debug(f'Scanned {X.shape[0]} rows and {X.shape[1]} columns from {len(files_h5)} .h5 files')
    else:
        df_embeddings = pl.read_parquet(f'{dir_sessions_embeddings_parquet}/*/*.parquet')
        X = np.array(df_embeddings['embedding'].to_list())
        sessions = df_embeddings['session']
        log.debug(f'Loaded {X.shape[0]} rows and {X.shape[1]} columns from .parquet files')

    log.info('Fit KMeans')

    res = []

    for n_clusters in tqdm(config.N_CLUSTERS_TO_FIND, unit='model'):
        if use_dask:
            log.debug(f'Init Dask KMeans with: n_clusters={n_clusters}')
            km = dask_ml.cluster.KMeans(
                n_clusters=n_clusters,
                max_iter=300,
                random_state=42,
            )
        else:
            log.debug(f'Init scikit KMeans with: n_clusters={n_clusters}')
            km = KMeans(
                init='random',
                n_clusters=n_clusters,
                n_init='auto',
                max_iter=300,
                random_state=42,
            )
        km.fit(X)

        log.info(f'KMeans: n_clusters={km.n_clusters}, inertia={km.inertia_:.2f}, n_iter={km.n_iter_}')
        res.append({'n_clusters': km.n_clusters, 'inertia': km.inertia_, 'n_iter': km.n_iter_})
        pd.DataFrame(res).to_csv(f'{dir_out}/logs.csv', index=False)

        # save clusters to disk
        df_clusters = pl.DataFrame({'session': sessions, 'cluster': np.array(km.labels_)},
                                   columns={'session': pl.Int32, 'cluster': pl.Int16})
        file_out = f'{dir_out}/sessions-clusters-{n_clusters}.parquet'
        df_clusters.write_parquet(file_out)

        if use_dask:
            dask_client.restart()

    dask_client.close(60)

# describe_numeric(df_sessions[['n']].to_pandas())
# df_sessions.filter(pl.col('session') == 4).to_pandas()
