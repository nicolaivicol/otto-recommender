import logging

import numpy as np
import polars as pl
import pandas as pd
import glob
import lightgbm

import config
from utils import set_display_options, describe_numeric

set_display_options()
log = logging.getLogger('rank.py')

# References:
# https://medium.datadriveninvestor.com/a-practical-guide-to-lambdamart-in-lightgbm-f16a57864f6
# https://github.com/jameslamb/lightgbm-dask-testing/blob/main/notebooks/demo.ipynb
# https://www.kaggle.com/code/radek1/polars-proof-of-concept-lgbm-ranker
# https://github.com/microsoft/LightGBM/blob/477cbf373ea2138a186568ac88ef221ac74c7c71/tests/python_package_test/test_dask.py#L480

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

data_split_alias = 'train-test'
dir_retrieved = f'{config.DIR_DATA}/{data_split_alias}-ltr'
files_retrieved = sorted(glob.glob(f'{dir_retrieved}/*.parquet'))

PARAMS_LGBM = {
    'objective': 'lambdarank',
    'boosting_type': 'gbdt',
    'metric': 'ndcg',
    'n_estimators': 300,
    'learning_rate': 0.05,
    'max_depth': 4,
    'num_leaves': 15,
    'colsample_bytree': 0.50,  # aka feature_fraction
    'subsample': 0.50,  # aka bagging_fraction
    # 'bagging_freq': 1,
    'min_child_samples': 1,  # aka min_data_in_leaf  ? read github link with test
    'importance_type': 'gain',
    'seed': 42,
}

# ******************************************************************************
import dask.dataframe as dd
from lightgbm import DaskLGBMRanker
from dask.distributed import Client, LocalCluster
from psutil import virtual_memory, cpu_count


if __name__ == "__main__":
    # **************************************************************************
    df_train = pl.read_parquet(files_retrieved[0])
    print(df_train.shape)
    # (12539090, 70)
    non_feats = ['session', 'aid_next', 'target_clicks', 'target_carts', 'target_orders']
    feats = [c for c in df_train.columns if c not in non_feats]
    X_train = df_train.select(feats).to_pandas().values
    y_train = df_train['target_clicks'].to_numpy()
    qids_train = df_train.groupby('session', maintain_order=True).agg([pl.count('aid_next').alias('n')])['n'].to_numpy()

    print(qids_train[:10])
    # array([ 43,  57,  82, 302, 112, 418, 192, 147, 289, 572], dtype=uint32)

    df_valid = pl.read_parquet(files_retrieved[1])
    print(df_valid.shape)
    # (11768357, 70)
    X_valid = df_valid.select(feats).to_pandas().values
    y_valid = df_valid['target_clicks'].to_numpy()
    qids_valid = df_valid.groupby('session', maintain_order=True).agg([pl.count('aid_next').alias('n')])['n'].to_numpy()


    # group_column
    # train.txt train.txt.query
    print('fit lgbm_ranker')
    lgbm_ranker = lightgbm.LGBMRanker(**PARAMS_LGBM)
    lgbm_ranker.fit(
        X=X_train,
        y=y_train,
        group=qids_train,
        feature_name=feats,
        eval_names=['valid', 'train'],
        eval_set=[(X_valid, y_valid), (X_train, y_train)],
        eval_group=[qids_valid, qids_train],
        eval_at=[20],
        # early_stopping_rounds=20,
        verbose=25,
    )
    # [25]	train's ndcg@5: 0.764674	train's ndcg@20: 0.794272
    # [50]	train's ndcg@5: 0.765991	train's ndcg@20: 0.795295

    pred = lgbm_ranker.predict(X_train)

    print(pred[:10])
    # array([-0.93658272, -1.05689281, -1.06906435,  0.00182868, -1.19103266,
    #        -0.25859635, -1.19103266, -0.93521062, -1.19103266, -0.93658272])


    pd.DataFrame({'pred': pred}).to_csv('pred.csv', index=False)

    df_featimp = pd.DataFrame(
        {'feat': feats, 'importance': lgbm_ranker.feature_importances_})\
        .sort_values('importance', ascending=False)
    print(df_featimp)
    # **************************************************************************

    memory_limit = round(virtual_memory().total / 1e9) - 1
    cluster = LocalCluster(
        n_workers=1,
        threads_per_worker=max(cpu_count() // 2, 1),
        memory_limit=f'{memory_limit}GB',
    )
    log.debug(f'Dask cluster dashboard link: {cluster.dashboard_link}')
    client = Client(cluster)
    log.debug(f'Dask client dashboard link: {client.dashboard_link}')

    # dd_train = dd.read_parquet(files_retrieved[0], engine='pyarrow')
    ddf_train = dd.read_parquet(
        path=files_retrieved[:1],
        # index='session',  # ['session', 'aid_next'],
        engine='pyarrow',
        chunksize='80mb',
        aggregate_files=True,
        ignore_metadata_file=True,
    )

    # separate target, weight from features.
    non_feats = ['session', 'aid_next', 'target_clicks', 'target_carts', 'target_orders']
    feats = [c for c in ddf_train.columns if c not in non_feats]
    dy = ddf_train['target_clicks']
    dX = ddf_train[feats]
    # dg = ddf_train.index.to_series()
    dg = ddf_train[['session']]

    # encode group identifiers into run-length encoding, the format LightGBMRanker is expecting
    # so that within each partition, sum(g) = n_samples.
    dg = dg.map_partitions(lambda p: p.groupby('session', sort=False).apply(lambda z: z.shape[0]))

    # print(dg.compute())
    # session
    # 11098528     43
    # 11098529     57
    # 11098530     82
    # 11098531    302
    # 11098532    112

    # ddf_train[['session', 'aid_next']].head(100)
    #      session  aid_next
    # 0   11098528    656322
    # 1   11098528   1390152
    # 2   11098528    205357
    # ...
    # 97  11098529      1453
    # 98  11098529   1383767
    # 99  11098529   1105029

    lgbm_ranker_dask = DaskLGBMRanker(
        time_out=5,
        tree_learner_type='data_parallel',
        client=client,
        **PARAMS_LGBM
    )

    print('fit lgbm_ranker_dask')
    lgbm_ranker_dask.fit(
        X=dX,
        y=dy,
        group=dg,
        feature_name=feats,
        eval_names=['train'],
        eval_set=[(dX, dy)],
        eval_group=[dg],
        eval_at=[5, 20],
        verbose=25,
    )
    # [25]	train's ndcg@5: 0.764674	train's ndcg@20: 0.794272
    # [50]	train's ndcg@5: 0.765991	train's ndcg@20: 0.795295

    pred_dask = lgbm_ranker_dask.predict(dX).compute()
    pd.DataFrame({'pred_dask': pred_dask}).to_csv('pred_dask.csv', index=False)

    # dask_ranker_local = lgbm_ranker_dask.to_local()
    # print(lgbm_ranker_dask.booster_.model_to_string())
    lgbm_ranker_dask.booster_.save_model('lgbm_ranker_dask.lgbm')

    client.close()

    # **************************************************************************
    print('compare')
    df_compare = pd.DataFrame({'pred_dask': pred_dask, 'pred': pred})
    print(df_compare.head(100))
    df_compare.to_csv('df_compare.csv', index=False)
    # **************************************************************************


    # ****************************

    raise Exception('here')




    file_candidates = files_retrieved[0]

    df_part = pl.read_parquet(file_candidates)
    # df_part.write_parquet(file_candidates, use_pyarrow=False)

    df_part = df_part.filter(~pl.col('n_uniq_aid').is_null()).sort(by=['session'])

    df_part = df_part.filter(~pl.col('n_uniq_aid').is_null()).sort(by=['session'])

    df_part.describe().to_pandas()

    non_feats = ['session', 'aid_next', 'target_clicks', 'target_carts', 'target_orders']
    feats = [c for c in df_part.columns if c not in non_feats]

    df_part = df_part.sample(10_000_000).sort(by=['session'])
    df_part.head(10).to_pandas()








    print(df_part.shape)

    df_part.filter(pl.col('session') == 11197056).to_pandas()[['session',  'aid_next', 'src_self', 'src_click_to_click',
    'src_click_to_cart_or_buy', 'src_cart_to_cart',  'src_buy_to_buy', 'src_w2vec_1_2', 'src_w2vec_all', 'rank_w2vec_all']]

    df_part.filter(pl.col('session') == 11197056).to_pandas()[
        ['n_w2vec_all', 'dist_w2vec_all', 'rank_w2vec_all', 'best_rank_w2vec_all', 'n_w2vec_1_2',
         'dist_w2vec_1_2', 'rank_w2vec_1_2', 'best_rank_w2vec_1_2',]]

