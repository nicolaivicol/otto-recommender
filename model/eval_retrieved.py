import logging
import argparse
import json
import polars as pl
import os.path

import config
from utils import get_submit_file_name


log = logging.getLogger(os.path.basename(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_split_alias', default='train-test')
    parser.add_argument('--tag', default=f'v{config.VERSION}')
    args = parser.parse_args()
    # python -m model.eval_retrieved

    log.info(f'Running {os.path.basename(__file__)} with parameters: \n' + json.dumps(vars(args), indent=2))
    log.info('This evaluates the retrieved candidates.')

    dir_retrieved = f'{config.DIR_DATA}/{args.data_split_alias}-retrieved/*.parquet'
    dir_labels = f'{config.DIR_DATA}/{args.data_split_alias}-parquet/test_labels/*.parquet'
    dir_out = f'{config.DIR_DATA}/{args.data_split_alias}-evals-retrieved'
    os.makedirs(dir_out)

    df_retrieved = pl.scan_parquet(dir_retrieved) \
        .select(['session', 'aid_next']) \
        .with_column(pl.lit(1).cast(pl.Int8).alias('submit')) \
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
            .groupby(['session']) \
            .agg([
                (pl.col('target') * pl.col('submit')).sum().cast(pl.Int16).alias('hit'),
                pl.sum('target').cast(pl.Int16).alias('true'),
                ]) \
            .with_columns([
                pl.col('hit').clip_max(200).alias('hit200'),
                pl.col('true').clip_max(200).alias('true200'),
                pl.col('hit').clip_max(100).alias('hit100'),
                pl.col('true').clip_max(100).alias('true100'),
                pl.col('hit').clip_max(20).alias('hit20'),
                pl.col('true').clip_max(20).alias('true20'),
                ]) \
            .sum() \
            .with_columns([
                (pl.col('hit20') / pl.col('true20')).alias('recall@20'),
                (pl.col('hit100') / pl.col('true100')).alias('recall@100'),
                (pl.col('hit200') / pl.col('true200')).alias('recall@200'),
                (pl.col('hit') / pl.col('true')).alias('recall@max'),
                ])

        lst_metrics.append(
            metrics \
                .select(['recall@20', 'recall@100', 'recall@200', 'recall@max']) \
                .with_column(pl.lit(type_int).cast(pl.Int64).alias('type_int'))
        )

    metrics = pl.concat(lst_metrics)

    metrics_agg = metrics \
        .join(pl.DataFrame({'type_int': [0, 1, 2], 'weight': [0.1, 0.3, 0.6]}), on='type_int') \
        .with_columns([
            pl.col('recall@20') * pl.col('weight'),
            pl.col('recall@100') * pl.col('weight'),
            pl.col('recall@200') * pl.col('weight'),
            pl.col('recall@max') * pl.col('weight'),
            ]) \
        .sum() \
        .drop('weight')

    metrics_all = pl.concat([metrics, metrics_agg]) \
        .join(pl.DataFrame({'type_int': [0, 1, 2, 3],
                            'type': ['clicks', 'carts', 'orders', 'total']}),
              on='type_int') \
        .select(['type', 'recall@20', 'recall@100', 'recall@200', 'recall@max'])

    log.info('Metrics: \n' + str(metrics_all))

    file_out = f'{dir_out}/{get_submit_file_name("eval-retrieved", args.tag)}.csv'
    metrics_all.write_csv(file_out, float_precision=4)

    log.info(f'Metrics saved to: {file_out}')
