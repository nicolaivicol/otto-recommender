import logging
import argparse
import json
import polars as pl
import pandas as pd
import os.path
from tabulate import tabulate

import config
from utils import get_submit_file_name, describe_numeric

log = logging.getLogger(os.path.basename(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_split_alias', default='train-test')
    parser.add_argument('--tag', default=f'v{config.VERSION}')
    parser.add_argument('--max_k', default=20)
    args = parser.parse_args()
    max_k = args.max_k
    # python -m model.eval_retrieved

    log.info(f'Running {os.path.basename(__file__)} with parameters: \n' + json.dumps(vars(args), indent=2))
    log.info('This evaluates the retrieved candidates. ')

    dir_retrieved = f'{config.DIR_DATA}/{args.data_split_alias}-retrieved/*.parquet'
    dir_labels = f'{config.DIR_DATA}/{args.data_split_alias}-parquet/test_labels/*.parquet'
    dir_out = f'{config.DIR_DATA}/{args.data_split_alias}-evals-retrieved'
    os.makedirs(dir_out, exist_ok=True)

    # dir_retrieved = '../data/train-test-retrieved/0000000_0100000.parquet'
    df_retrieved = pl.scan_parquet(dir_retrieved) \
        .select(['session', 'aid_next']) \
        .with_column(pl.lit(1).cast(pl.Int8).alias('submit')) \
        .with_column((pl.col('aid_next').cumcount().over('session') + 1).cast(pl.Int16).alias('rank')) \
        .collect()

    df_labels = pl.read_parquet(dir_labels) \
        .with_column(pl.lit(1).cast(pl.Int8).alias('target'))

    lst_metrics = []

    for type_int in [0, 1, 2]:
        log.debug(f'evaluating type_int={type_int}')
        df = df_retrieved \
            .join(df_labels.filter(pl.col('type') == type_int).drop('type'),
                  left_on=['session', 'aid_next'],
                  right_on=['session', 'aid'],
                  how='outer') \
            .fill_null(0)

        metrics = df \
            .with_column((pl.col('target') * pl.col('submit')).cast(pl.Int8).alias('hit')) \
            .with_columns([
                (pl.col('hit') * (pl.col('rank') <= 200)).alias('hit@200'),
                (pl.col('hit') * (pl.col('rank') <= 100)).alias('hit@100'),
                (pl.col('hit') * (pl.col('rank') <= 20)).alias('hit@20'),
                ]) \
            .groupby(['session']) \
            .agg([
                pl.sum('hit@20').clip_max(max_k).cast(pl.Int16).alias('hit@20'),
                pl.sum('hit@100').clip_max(max_k).cast(pl.Int16).alias('hit@100'),
                pl.sum('hit@200').clip_max(max_k).cast(pl.Int16).alias('hit@200'),
                pl.sum('hit').clip_max(max_k).cast(pl.Int16).alias('hit@max'),
                pl.sum('target').clip_max(max_k).cast(pl.Int16).alias('true'),
                ]) \
            .sum() \
            .with_columns([
                (pl.col('hit@20') / pl.col('true')).alias('top20'),
                (pl.col('hit@100') / pl.col('true')).alias('top100'),
                (pl.col('hit@200') / pl.col('true')).alias('top200'),
                (pl.col('hit@max') / pl.col('true')).alias('topall'),
                ])

        lst_metrics.append(
            metrics \
                .select(['top20', 'top100', 'top200', 'topall']) \
                .with_column(pl.lit(type_int).cast(pl.Int64).alias('type_int'))
        )

    metrics = pl.concat(lst_metrics)

    metrics_agg = metrics \
        .join(pl.DataFrame({'type_int': [0, 1, 2], 'weight': [0.1, 0.3, 0.6]}), on='type_int') \
        .with_columns([
            pl.col('top20') * pl.col('weight'),
            pl.col('top100') * pl.col('weight'),
            pl.col('top200') * pl.col('weight'),
            pl.col('topall') * pl.col('weight'),
            ]) \
        .sum() \
        .drop('weight')

    metrics_all = pl.concat([metrics, metrics_agg]) \
        .join(pl.DataFrame({'type_int': [0, 1, 2, 3],
                            'type': ['clicks', 'carts', 'orders', 'total']}),
              on='type_int') \
        .select(['type', 'top20', 'top100', 'top200', 'topall']) \
        .rename({k: f'recall@{max_k}-{k}' for k in ['top20', 'top100', 'top200', 'topall']})

    log.info(f'Maximum recal@{max_k} possible for top K retrieved candidates if ranked ideally: \n' + str(metrics_all))

    file_out = f'{dir_out}/{get_submit_file_name("eval-retrieved", args.tag)}.csv'
    metrics_all.write_csv(file_out, float_precision=4)

    log.info(f'Metrics saved to: {file_out}')

    srcs = ['src_any', 'src_self', 'src_click_to_click', 'src_click_to_cart_or_buy', 'src_cart_to_cart',
            'src_cart_to_buy', 'src_buy_to_buy', 'src_w2vec_all', 'src_w2vec_1_2']
    df_retrieved = pl.scan_parquet(dir_retrieved).select(['session'] + srcs).collect()

    lst_summary = []
    for src in srcs:
        stats_summary = describe_numeric(df_retrieved.groupby('session').agg([pl.sum(src).alias('n')]).to_pandas()[['n']])
        lst_summary.append(pd.concat([pd.DataFrame({'source': [src]}), stats_summary.reset_index(drop=True)], axis=1, ignore_index=False))

    stats_summary = pd.concat(lst_summary).drop(columns=['count', 'std', 'count_nan', 'prc_nan'])
    log.info(f'Stats of number of aids per session: \n{str(tabulate(stats_summary, headers=stats_summary.columns, showindex=False))}')



# ******************************************************************************
# V.1.0.0
# Maximum recal@20 possible for top K candidates:
# ┌────────┬─────────────────┬──────────────────┬──────────────────┬──────────────────┐
# │ type   ┆ recall@20-top20 ┆ recall@20-top100 ┆ recall@20-top200 ┆ recall@20-topall │
# │ ---    ┆ ---             ┆ ---              ┆ ---              ┆ ---              │
# │ str    ┆ f64             ┆ f64              ┆ f64              ┆ f64              │
# ╞════════╪═════════════════╪══════════════════╪══════════════════╪══════════════════╡
# │ clicks ┆ 0.174403        ┆ 0.47164          ┆ 0.522317         ┆ 0.550825         │
# │ carts  ┆ 0.10698         ┆ 0.340022         ┆ 0.417022         ┆ 0.494018         │
# │ orders ┆ 0.111423        ┆ 0.386607         ┆ 0.516885         ┆ 0.704218         │
# │ total  ┆ 0.116388        ┆ 0.381135         ┆ 0.487469         ┆ 0.625819         │
# └────────┴─────────────────┴──────────────────┴──────────────────┴──────────────────┘
# Stats of number of aids per session:
#       count     mean      std    min    1%    5%    10%    25%    50%    75%    90%    95%    99%    max
# -----------  -------  -------  -----  ----  ----  -----  -----  -----  -----  -----  -----  -----  -----
# 1.78374e+06  113.306  142.586      1     1    20     32     48     61    121    239    368    789   3994
#
# V.1.1.0 retrieve more co-counts
# ┌────────┬─────────────────┬──────────────────┬──────────────────┬──────────────────┐
# │ type   ┆ recall@20-top20 ┆ recall@20-top100 ┆ recall@20-top200 ┆ recall@20-topall │
# │ ---    ┆ ---             ┆ ---              ┆ ---              ┆ ---              │
# │ str    ┆ f64             ┆ f64              ┆ f64              ┆ f64              │
# ╞════════╪═════════════════╪══════════════════╪══════════════════╪══════════════════╡
# │ clicks ┆ 0.196172        ┆ 0.522423         ┆ 0.547957         ┆ 0.556023         │
# │ carts  ┆ 0.152472        ┆ 0.421397         ┆ 0.462483         ┆ 0.4998           │
# │ orders ┆ 0.160223        ┆ 0.48024          ┆ 0.581966         ┆ 0.709191         │
# │ total  ┆ 0.161492        ┆ 0.466805         ┆ 0.54272          ┆ 0.631057         │
# └────────┴─────────────────┴──────────────────┴──────────────────┴──────────────────┘
# Stats of number of aids per session, by type:
# source                          mean    min    1%    5%    10%    25%    50%    75%    90%    95%    99%    max
# ------------------------  ----------  -----  ----  ----  -----  -----  -----  -----  -----  -----  -----  -----
# src_any                   120.274         1     1    20     34     54     68    135    258    379    698   2333
# src_self                    3.06404       1     1     1      1      1      1      3      6     10     24    137
# src_click_to_click         20.4743        0     0     2      9     10     10     21     41     63    133    553
# src_click_to_cart_or_buy   20.1769        0     0     0      3     10     10     20     42     66    139    577
# src_cart_to_cart            4.6614        0     0     0      0      0      0      0     20     29     80    669
# src_cart_to_buy             3.31474       0     0     0      0      0      0      0      7     22     63    509
# src_buy_to_buy              0.324095      0     0     0      0      0      0      0      0      0      9    262
# src_w2vec_all              42.785         0     0     0     20     20     20     48     91    132    242    881
# src_w2vec_1_2              40.9392        0     0     0      0     20     20     41     92    133    242    871
