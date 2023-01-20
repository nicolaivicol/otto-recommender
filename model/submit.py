import logging
import argparse
import json
import polars as pl
import os.path

import config
from utils import get_submit_file_name


log = logging.getLogger(os.path.basename(__file__))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_split_alias', default='train-test')
    parser.add_argument('--keep_top_k', type=int, default=20)
    parser.add_argument('--tag', default=f'v{config.VERSION}')
    args = parser.parse_args()
    # python -m model.submit --data_split_alias full

    log.info(f'Running {os.path.basename(__file__)} with parameters: \n' + json.dumps(vars(args), indent=2))
    log.info('This generates a submission for kaggle.')

    dir_ranked = f'{config.DIR_DATA}/{args.data_split_alias}-ranked'
    dir_out = f'{config.DIR_DATA}/{args.data_split_alias}-submit'
    os.makedirs(dir_out, exist_ok=True)

    df_submit = []
    recalls = {}

    for target in ['clicks', 'carts', 'orders']:
        log.debug(f'target={target}')
        df_pred = pl.read_parquet(f'{dir_ranked}/{target}/*.parquet')

        df_pred = df_pred\
            .select(['session', 'aid_next', 'pred_score']) \
            .with_column(pl.col('pred_score').rank('ordinal', reverse=True)
                         .over('session').cast(pl.Int16).alias('pred_rank')) \
            .filter((pl.col('pred_rank') <= args.keep_top_k)) \
            .sort(['session', 'pred_rank'])

        log.debug('\n' + str(df_pred.head(10)))

        df_pred = df_pred.with_column(pl.lit(target).alias('type')) \
            .with_column(pl.format('{}_{}', 'session', 'type').alias('session_type')) \
            .groupby('session_type') \
            .agg([pl.col('aid_next').cast(str).list().alias('labels')]) \
            .with_column(pl.col('labels').arr.join(' '))\
            .sort('session_type')

        log.debug('\n' + str(df_pred.head(10)))

        df_submit.append(df_pred)

    df_submit = pl.concat(df_submit).sort('session_type')
    log.debug('\n' + str(df_submit.head(10)))

    file_out = f'{dir_out}/{get_submit_file_name(args.tag)}.csv'
    df_submit.write_csv(file_out, has_header=True)
    log.info(f'submission with {df_submit.shape[0]} rows saved to {file_out}')
