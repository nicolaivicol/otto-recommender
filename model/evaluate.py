import logging
import argparse
import json
import polars as pl
import os.path

import config

log = logging.getLogger(os.path.basename(__file__))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_split_alias', default='train-test')
    parser.add_argument('--file_submit', default='submission-v1.0.0-7fa08333-20230119143255.csv')
    args = parser.parse_args()

    log.info(f'Running {os.path.basename(__file__)} with parameters: \n' + json.dumps(vars(args), indent=2))
    log.info('This evaluates a submission on test data.')

    labels = pl.read_parquet(f'{config.DIR_DATA}/{args.data_split_alias}-parquet/test_labels/*.parquet')
    submission = pl.read_csv(f'{config.DIR_DATA}/{args.data_split_alias}-submit/{args.file_submit}')
    dir_out = f'{config.DIR_DATA}/{args.data_split_alias}-evaluate'
    os.makedirs(dir_out, exist_ok=True)

    labels = labels \
        .rename({'type': 'type_int'}). \
        join(pl.DataFrame({'type_int': [0, 1, 2], 'type': ['clicks', 'carts', 'orders']})
             .with_column(pl.col('type_int').cast(pl.Int8)), on='type_int') \
        .drop('type_int') \
        .with_column(pl.lit(1).alias('target'))

    submission = submission \
        .with_column(pl.col('session_type').str.split('_').alias('session_type_split')) \
        .with_column(pl.col('session_type_split').arr.get(0).alias('session').cast(pl.Int32)) \
        .with_column(pl.col('session_type_split').arr.get(1).alias('type')) \
        .with_column(pl.col('labels').str.split(' ')) \
        .explode('labels') \
        .with_column(pl.col('labels').cast(pl.Int32).alias('aid')) \
        .drop(['labels', 'session_type', 'session_type_split']) \
        .with_column(pl.lit(1).alias('submit'))

    joined = labels.join(submission, on=['session', 'type', 'aid'], how='outer').fill_null(0)

    joined = joined \
        .groupby(['session', 'type']) \
        .agg([pl.sum('target').clip_max(20).alias('true'),
              (pl.col('target') * pl.col('submit')).sum().alias('hit')]) \
        .groupby('type') \
        .agg([pl.sum('hit'), pl.sum('true')]) \
        .with_column((pl.col('hit') / pl.col('true')).alias('recall@20'))

    log.debug('stats per type: \n' + str(joined))

    recall_agg = joined \
        .join(pl.DataFrame({'type': ['clicks', 'carts', 'orders'], 'weight': [0.1, 0.3, 0.6]}), on='type') \
        .with_column(pl.col('recall@20') * pl.col('weight')) \
        .sum()

    res = dict(zip(joined['type'], joined['recall@20']))
    res['total'] = recall_agg[0, 'recall@20']

    log.info(f'Recall@20: {json.dumps(res)}')

    with open(f'{dir_out}/{args.file_submit}.json', 'w') as f:
        json.dump(res, f)

    # submission-v1.0.0-7fa08333-20230119143255.csv
    # Recall@20: {"clicks": 0.49250158806146027, "orders": 0.650877277624998, "carts": 0.40742088481818745, "total": 0.562002790826601}
    # ┌────────┬────────┬─────────┬───────────┐
    # │ type   ┆ hit    ┆ true    ┆ recall@20 │
    # │ ---    ┆ ---    ┆ ---     ┆ ---       │
    # │ str    ┆ i32    ┆ i32     ┆ f64       │
    # ╞════════╪════════╪═════════╪═══════════╡
    # │ carts  ┆ 230643 ┆ 566105  ┆ 0.407421  │
    # │ clicks ┆ 855952 ┆ 1737968 ┆ 0.492502  │
    # │ orders ┆ 202361 ┆ 310905  ┆ 0.650877  │
    # └────────┴────────┴─────────┴───────────┘
