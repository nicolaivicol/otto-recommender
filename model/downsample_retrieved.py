import polars as pl
import argparse
import glob
import os
import json
import logging
from tqdm import tqdm

import config

log = logging.getLogger(os.path.basename(__file__))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_split_alias', default='train-test')
    parser.add_argument('--keep_ratio_neg_to_pos', default=config.DOWNSAMPLE_RATIO_NEG_TO_POS, type=float)
    parser.add_argument('--max_neg_per_session', default=config.DOWNSAMPLE_MAX_NEG_PER_SESSION, type=float)
    args = parser.parse_args()
    keep_ratio_neg_to_pos = args.keep_ratio_neg_to_pos
    max_neg_per_session = args.max_neg_per_session

    log.info(f'Running {os.path.basename(__file__)} with parameters: \n' + json.dumps(vars(args), indent=2))
    log.info(f'This downsamples the retrieved candidates. ETA 3-5min. \n'
             f'Will keep a max ratio of {keep_ratio_neg_to_pos} negative to positive samples (negative downsampling), '
             f'but maximum {max_neg_per_session} negative samples per session.')

    # python -m model.downsample_retrieved

    dir_retrieved_w_feats = f'{config.DIR_DATA}/{args.data_split_alias}-retrieved'
    files = sorted(glob.glob(f'{dir_retrieved_w_feats}/*.parquet'))
    dir_out = f'{config.DIR_DATA}/{args.data_split_alias}-retrieved-downsampled'

    for file in tqdm(files, unit='file', leave=False):
        df = pl.read_parquet(file)

        for target in config.TYPES:
            # filter out sessions without positive samples for this target type
            df_target = df \
                .with_column(pl.sum(f'target_{target}').over('session').alias('n_pos')) \
                .filter(pl.col('n_pos') > 0)

            # drop other target columns
            df_target = df_target.drop([f'target_{t}' for t in config.TYPES if t != target])

            # downsample negative samples to keep a ratio of keep_ratio_neg_to_pos of negative to positive samples
            df_target = df_target \
                .with_column((pl.col('n_pos') * keep_ratio_neg_to_pos)
                             .clip_max(max_neg_per_session)
                             .alias('max_n_neg'))
            df_pos = df_target.filter(pl.col(f'target_{target}') == 1)
            df_neg = df_target.filter(pl.col(f'target_{target}') == 0) \
                .with_column(pl.arange(0, pl.count()).shuffle(seed=42).over('session').alias('random')) \
                .filter(pl.col('random') < pl.col('max_n_neg')) \
                .drop('random')
            df_target = pl.concat([df_pos, df_neg]) \
                .drop(['max_n_neg', 'n_pos']) \
                .sort(['session', f'target_{target}'], reverse=[False, True])

            # save data separately for each target type
            os.makedirs(f'{dir_out}/{target}', exist_ok=True)
            df_target.write_parquet(f'{dir_out}/{target}/{os.path.basename(file)}')

            avg_nr_candidates_per_session = df_target.groupby('session').agg(pl.count()).mean()[0,1]
            avg_nr_pos_per_session = df_target.groupby('session').agg(pl.sum(f'target_{target}')).mean()[0, 1]

            log.info(f'Downsampled file {os.path.basename(file)} for target={target} '
                     f'to {len(df_target):,} rows, which is {len(df_target) / len(df) * 100:.2f}% of original size. \n'
                     f'Average nr of candidates per session: {avg_nr_candidates_per_session:.2f} \n'
                     f'Average nr of positive samples per session: {avg_nr_pos_per_session:.2f}')

    log.info('Done.')
