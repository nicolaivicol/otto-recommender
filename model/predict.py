import logging
import argparse
import json
import numpy as np
import polars as pl
import glob
import os.path
from tqdm import tqdm
from typing import List, Dict, Union

import config
from utils import set_display_options, compute_recall_at_k
from lgbm_rankers import load_lgbm, get_file_name, load_data_for_lgbm_predict

set_display_options()
log = logging.getLogger(os.path.basename(__file__))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_split_alias', default='train-test')
    parser.add_argument('--targets', nargs='+', default=['clicks', 'carts', 'orders'])
    parser.add_argument('--keep_top_k', type=int, default=20)
    args = parser.parse_args()

    log.info(f'Running {os.path.basename(__file__)} with parameters: \n' + json.dumps(vars(args), indent=2))
    log.info('This predicts scores for the retrieved AIDs. ETA 30min.')

    dir_models = f'{config.DIR_ARTIFACTS}/lgbm'
    dir_out = f'{config.DIR_DATA}/{args.data_split_alias}-predict'
    os.makedirs(dir_out, exist_ok=True)
    dir_retrieved_w_feats = f'{config.DIR_DATA}/{args.data_split_alias}-retrieved'
    data_files = sorted(glob.glob(f'{dir_retrieved_w_feats}/*.parquet'))

    pb = tqdm(desc='pred > rank > save', total=len(args.targets)*len(data_files), unit='file')

    for target in args.targets:
        log.info(f'predict scores for target={target} \n')
        # target = 'orders'
        model_file = get_file_name(dir_models, target, args.data_split_alias)
        lgbm_ranker = load_lgbm(model_file)
        feat_names = lgbm_ranker.feature_name()
        dir_out_target = f'{dir_out}/{target}'
        os.makedirs(dir_out_target, exist_ok=True)

        for data_file in data_files:
            log.debug(f'predict scores for file={os.path.basename(data_file)} \n')
            # data_file = data_files[17]
            X, session, aid, is_retrieved, y = load_data_for_lgbm_predict(data_file, feat_names, f'target_{target}')
            pred_score = lgbm_ranker.predict(X)

            # recall@20 = 0.57 (0.1*0.49 + 0.3*0.42 + 0.6*0.65)
            if y is not None:
                recall_20 = compute_recall_at_k(session, y, pred_score, is_retrieved, k=20)
                log.debug(f'recall@20={recall_20:.4}')

            cols = {'session': session, 'aid': aid, 'pred_score': pred_score, 'is_retrieved': is_retrieved}
            if y is not None:
                cols[f'target_{target}'] = y

            df_pred = pl.DataFrame(cols) \
                .select([
                    pl.all(),
                    (pl.col('pred_score').rank('ordinal', reverse=True)
                     .over('session').cast(pl.Int16).alias('pred_rank'))]) \
                .sort(['session', 'pred_rank'])

            if args.keep_top_k is not None:
                df_pred = df_pred.filter((pl.col('pred_rank') <= args.keep_top_k) | (pl.col('is_retrieved') == 0))

            df_pred.write_parquet(f'{dir_out_target}/{os.path.basename(data_file)}')

            pb.update()
