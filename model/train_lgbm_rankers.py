import logging
import argparse
import json
import numpy as np
import polars as pl
import pandas as pd
import glob
import os.path
import pickle
from typing import List, Union
import lightgbm
import dask.dataframe as dd
from dask.distributed import Client

import config
from dask_utils import set_up_dask_client
from utils import set_display_options, get_best_metric, get_best_iter

set_display_options()
log = logging.getLogger(os.path.basename(__file__))

# Statistics:
# number of candidates per session:
# describe_numeric(df_part.groupby('session').agg([pl.count('aid_next').alias('n')]).to_pandas()[['n']])
# ver 3
#    mean     std   min     5%    25%    50%     75%     95%      max
# 114.877 144.342 1.000 20.000 46.000 56.000 125.000 398.000 1373.000
# ver 4
#    mean     std   min     5%    25%    50%     75%     95%      max
# 126.015 161.630 1.000 21.000 49.000 66.000 135.000 430.000 2500.000

# positive target rate:
# df_part.select(pl.col(['target_clicks', 'target_carts', 'target_orders'])).mean().to_pandas()
#  target_clicks | target_carts | target_orders
#  0.004535      | 0.001491     | 0.00122


def _infer_feats_from_df(df):
    non_feats = ['session', 'aid_next', 'target_clicks', 'target_carts', 'target_orders', 'rank_total_cl1', ]
    return [c for c in df.columns if c not in non_feats]


def load_data_for_lgbm_standard(source: Union[str, List[str]], target: str, feats: List[str] = None):
    if isinstance(source, List):
        df = pl.concat([pl.read_parquet(s) for s in source])
    else:
        df = pl.read_parquet(source)

    if feats is None:
        feats = _infer_feats_from_df(df)

    # df = df.with_column(pl.col([f'target_{type}' for type in config.TYPES]).fill_null(0).clip_min(0))

    X = df.select(feats).to_pandas().values
    y = df[target].to_numpy()
    group_counts = df.groupby('session', maintain_order=True).agg([pl.count('aid_next').alias('n')])['n'].to_numpy()

    log.debug('X.shape: %s, y.shape: %s, group_counts.shape: %s', X.shape, y.shape, group_counts.shape)

    return X, y, group_counts, feats


def load_data_for_lgbm_predict(file: str, feats: List[str]):
    df = pl.read_parquet(file)
    X = df.select(feats).to_pandas().values
    session = df['session'].to_numpy()
    aid_next = df['aid_next'].to_numpy()
    return X, session, aid_next


def load_data_for_lgbm_dask(source: Union[str, List[str]], target: str, feats: List[str] = None):
    ddf = dd.read_parquet(
        path=source,
        engine='pyarrow',
        chunksize='100Mb',
        aggregate_files=True,
        ignore_metadata_file=True,
    )

    if feats is None:
        feats = _infer_feats_from_df(ddf)

    X = ddf[feats]
    y = ddf[target]

    # encode group identifiers into run-length encoding, the format LightGBMRanker is expecting
    # so that within each partition, sum(g) = n_samples.
    group_counts = ddf[['session']].map_partitions(lambda p: p.groupby('session', sort=False).apply(lambda z: z.shape[0]))

    return X, y, group_counts, feats


def load_data_for_lgbm(source: Union[str, List[str]], target: str, feats: List[str] = None, dask_client: Client = None):
    if dask_client is not None:
        return load_data_for_lgbm_dask(source, target, feats)
    else:
        return load_data_for_lgbm_standard(source, target, feats)


def _handle_evals(X_train, y_train, group_train, X_valid=None, y_valid=None, group_valid=None):
    valid_data_exists = X_valid is not None and y_valid is not None and group_valid is not None
    if valid_data_exists:
        eval_names, eval_set, eval_group = (['valid', 'train'], [(X_valid, y_valid), (X_train, y_train)], [group_valid, group_train])
    else:
        eval_names, eval_set, eval_group = ['train'], [(X_train, y_train)], [group_train]

    return eval_names, eval_set, eval_group


def fit_lgbm(X_train, y_train, group_train, feats, X_valid=None, y_valid=None, group_valid=None, dask_client: Client = None):
    eval_names, eval_set, eval_group = _handle_evals(X_train, y_train, group_train, X_valid, y_valid, group_valid)

    if dask_client is not None:
        lgbm_ranker = lightgbm.DaskLGBMRanker(tree_learner_type='data_parallel', client=dask_client, time_out=5, **config.PARAMS_LGBM)
    else:
        lgbm_ranker = lightgbm.LGBMRanker(**config.PARAMS_LGBM)

    log.debug(f'fit {type(lgbm_ranker)} with params:\n{json.dumps(config.PARAMS_LGBM, indent=2)}')
    lgbm_ranker.fit(
        X=X_train,
        y=y_train,
        group=group_train,
        feature_name=feats,
        eval_names=eval_names,
        eval_set=eval_set,
        eval_group=eval_group,
        **config.PARAMS_LGBM_FIT,
    )
    return lgbm_ranker


def feature_importance_lgbm(
        lgbm_model: Union[lightgbm.LGBMRanker, lightgbm.DaskLGBMRanker],
        feature_names: List[str],
        importance_type='gain'
) -> pd.DataFrame:
    try:
        feat_imp = lgbm_model.feature_importance(importance_type=importance_type)
    except:
        feat_imp = lgbm_model.feature_importances_
    feat_imp = list(np.round(feat_imp / feat_imp.sum(), 4))
    feat_imp = pd.DataFrame({'feature': feature_names, 'importance': feat_imp})
    feat_imp = feat_imp.sort_values(by=['importance'], ascending=False)
    return feat_imp


def save_lgbm(lgbm_model: Union[lightgbm.LGBMRanker, lightgbm.DaskLGBMRanker], file_name: str):
    if isinstance(lgbm_model, lightgbm.DaskLGBMRanker):
        with open(f'{file_name}.dasklgbm.pickle', "wb") as f:
            pickle.dump(lgbm_model, f)
        log.debug(f'Dask estimator saved to: {file_name}.dasklgbm.pickle')
        lgbm_model = lgbm_model.to_local()  # convert to local estimator (sklearn equivalent)

    with open(f'{file_name}.lgbm.pickle', "wb") as f:
        pickle.dump(lgbm_model, f)
    log.debug(f'Estimator saved to: {file_name}.lgbm.pickle')

    lgbm_model.booster_.save_model(f'{file_name}.booster.lgbm', importance_type='gain')
    log.debug(f'Booster saved to: {file_name}.booster.lgbm')


def load_lgbm(file_name, format: str = 'booster.lgbm') -> Union[lightgbm.LGBMRanker, lightgbm.DaskLGBMRanker, lightgbm.Booster]:
    model_file = f'{file_name}.{format}'

    if format in ['dasklgbm.pickle', 'lgbm.pickle']:
        with open(model_file, 'rb') as f:
            lgbm_model = pickle.load(f)
    elif format == 'booster.lgbm':
        lgbm_model = lightgbm.Booster(model_file=model_file)
    else:
        raise ValueError(f"Unrecognized format={format}, format must be one of: "
                         f"'dasklgbm.pickle', 'lgbm.pickle', 'booster.lgbm'")

    return lgbm_model


def get_file_name(dir_out, target, *args):
    file_name = f'{dir_out}/{target}'
    if args:
        file_name += '-' + '-'.join(args)
    return file_name


def split_files_to_train_valid(files, valid_frac, max_files_in_train, max_files_in_valid):
    if valid_frac > 0:
        assert valid_frac < 1, 'valid_frac must be < 1'
        assert len(files) >= 2, 'need at least 2 files for train/test split'
        n_train = min(int((1 - valid_frac) * len(files)), len(files) - 1)
        files_train, files_valid = files[:n_train], files[n_train:]
    else:
        files_train, files_valid = files, None

    if max_files_in_train is not None:
        files_train = files_train[:int(max_files_in_train)]

    if max_files_in_valid is not None and files_valid is not None:
        files_valid = files_valid[:int(max_files_in_valid)]

    log.debug(f'{len(files_train)} files selected for train')

    if files_valid is not None:
        log.debug(f'{len(files_valid)} files selected for validation')

    return files_train, files_valid


def save_feat_imp(lgbm_model, feature_names, file_name):
    feat_imp = feature_importance_lgbm(lgbm_model, feature_names)
    feat_imp.to_csv(f'{file_name}-featimp.csv', index=False)
    return f'{file_name}-featimp.csv'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_split_alias', default='train-test')
    parser.add_argument('--valid_frac', default=0, type=float)  # 0: runs on full data, without validation, 0.30 - 30%
    parser.add_argument('--targets', nargs='+', default=['clicks', 'carts', 'orders'])
    parser.add_argument('--use_dask', default=0, type=int)
    parser.add_argument('--max_files_in_train', type=int)
    parser.add_argument('--max_files_in_valid', type=int)
    args = parser.parse_args()

    # python -m model.train_lgbm_rankers --valid_frac 0.25 --use_dask 0 --max_files_in_train 6 --max_files_in_valid 1

    log.info(f'Running {os.path.basename(__file__)} with parameters: \n' + json.dumps(vars(args), indent=2))
    log.info('This trains ranker models for clicks/carts/orders. ETA 5-10min.')

    dir_retrieved_w_feats = f'{config.DIR_DATA}/{args.data_split_alias}-retrieved-downsampled'
    dir_out = f'{config.DIR_ARTIFACTS}/lgbm'
    os.makedirs(dir_out, exist_ok=True)

    dask_client = set_up_dask_client() if args.use_dask == 1 else None

    for target in args.targets:
        log.info(f'Train LightGBM model for target={target}')

        log.debug('Split data for training and load it...')
        files = sorted(glob.glob(f'{dir_retrieved_w_feats}/{target}/*.parquet'))
        files_train, files_valid = split_files_to_train_valid(files, args.valid_frac, args.max_files_in_train, args.max_files_in_valid)
        X_train, y_train, group_train, feats = load_data_for_lgbm(files_train, f'target_{target}', dask_client=dask_client)
        X_valid, y_valid, group_valid = None, None, None
        if files_valid is not None:
            X_valid, y_valid, group_valid, _ = load_data_for_lgbm(files_valid, f'target_{target}', dask_client=dask_client)

        log.debug('Training...')
        lgbm_ranker = fit_lgbm(X_train, y_train, group_train, feats, X_valid, y_valid, group_valid, dask_client)

        metric_, best_score = get_best_metric(lgbm_ranker)
        best_iter = get_best_iter(lgbm_ranker)
        log.debug(f'Model finished training. Best metric: {metric_}={best_score:.4f}, best iteration={best_iter}')

        feat_imp = feature_importance_lgbm(lgbm_ranker, feats)
        log.debug(f'Feature importance: \n{feat_imp.head(25)}')

        log.debug('Save model to disk...')
        file_name = get_file_name(dir_out, target)
        save_lgbm(lgbm_ranker, file_name)
        # save_feat_imp(lgbm_ranker, feats, file_name)

    if dask_client is not None:
        dask_client.close(60)

    log.info(f'{os.path.basename(__file__)} ran successfully.')
