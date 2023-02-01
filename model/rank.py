import logging
import argparse
import json
import polars as pl
import glob
import os.path
from tqdm import tqdm

import config
from utils import set_display_options
from model.train_lgbm_rankers import load_lgbm, get_file_name, load_data_for_lgbm_predict

set_display_options()
log = logging.getLogger(os.path.basename(__file__))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_split_alias', default='train-test')
    parser.add_argument('--targets', nargs='+', default=['clicks', 'carts', 'orders'])
    parser.add_argument('--keep_top_k', type=int, default=config.KEEP_TOP_K)
    args = parser.parse_args()

    # python -m model.rank --data_split_alias full

    log.info(f'Running {os.path.basename(__file__)} with parameters: \n' + json.dumps(vars(args), indent=2))
    log.info('This predicts scores for the retrieved AIDs. ETA 60min.')

    dir_models = f'{config.DIR_ARTIFACTS}/lgbm'
    dir_out = f'{config.DIR_DATA}/{args.data_split_alias}-ranked'
    os.makedirs(dir_out, exist_ok=True)
    dir_retrieved_w_feats = f'{config.DIR_DATA}/{args.data_split_alias}-retrieved'
    data_files = sorted(glob.glob(f'{dir_retrieved_w_feats}/*.parquet'))

    pb = tqdm(desc='pred > rank > save', total=len(args.targets)*len(data_files), unit='file')

    for target in args.targets:
        log.info(f'predict scores for target={target}')

        model_file = get_file_name(dir_models, target)
        lgbm_ranker = load_lgbm(model_file)
        feat_names = lgbm_ranker.feature_name()
        dir_out_target = f'{dir_out}/{target}'
        os.makedirs(dir_out_target, exist_ok=True)

        for data_file in data_files:
            log.debug(f'predict scores for file={os.path.basename(data_file)}')

            X, session, aid_next = load_data_for_lgbm_predict(data_file, feat_names)
            pred_score = lgbm_ranker.predict(X)

            df_pred = pl.DataFrame({'session': session, 'aid_next': aid_next, 'pred_score': pred_score})
            df_pred = df_pred \
                .with_column((pl.col('pred_score').rank('ordinal', reverse=True)
                              .over('session').cast(pl.Int16).alias('pred_rank'))) \
                .filter((pl.col('pred_rank') <= args.keep_top_k)) \
                .sort(['session', 'pred_rank'])

            df_pred.write_parquet(f'{dir_out_target}/{os.path.basename(data_file)}')

            pb.update()
