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
    log.info('This evaluates the retrieved candidates. ETA 15min.')

    dir_retrieved = f'{config.DIR_DATA}/{args.data_split_alias}-retrieved/*.parquet'
    dir_labels = f'{config.DIR_DATA}/{args.data_split_alias}-parquet/test_labels/*.parquet'
    dir_out = f'{config.DIR_DATA}/{args.data_split_alias}-eval-retrieved'
    os.makedirs(dir_out, exist_ok=True)

    df_labels = pl.read_parquet(dir_labels) \
        .with_column(pl.lit(1).cast(pl.Int8).alias('target'))

    lst_metrics_all = []

    srcs = ['src_any', 'src_self', 'src_click_to_click', 'src_click_to_cart_or_buy', 'src_cart_to_cart',
            'src_cart_to_buy', 'src_buy_to_buy', 'src_w2vec_all', 'src_w2vec_1_2', 'src_pop_cl50', ]

    filters_src = {src: pl.col(src) == 1 for src in srcs}
    filters_src_not_self = {f'{src} & not self_src': ((pl.col(src) == 1) & (pl.col('src_self') == 0))
                            for src in srcs if src not in ['src_any', 'src_self']}
    filters = {**filters_src, **filters_src_not_self}

    for src, src_filter in filters.items():
        log.debug(f'evaluating src={src}')
        # dir_retrieved = '../data/train-test-retrieved/0000000_0100000.parquet'
        df_retrieved = pl.scan_parquet(dir_retrieved) \
            .filter(src_filter) \
            .select(['session', 'aid_next']) \
            .with_column(pl.lit(1).cast(pl.Int8).alias('submit')) \
            .with_column((pl.col('aid_next').cumcount().over('session') + 1).cast(pl.Int16).alias('rank')) \
            .collect()

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
                  on='type_int')\
            .with_column(pl.lit(src).alias('source')) \
            .select(['source', 'type', 'top20', 'top100', 'top200', 'topall']) \
            .rename({k: f'recall@{max_k}-{k}' for k in ['top20', 'top100', 'top200', 'topall']})

        lst_metrics_all.append(metrics_all)

    metrics_all = pl.concat(lst_metrics_all)

    log.info(f'The maximum recal@{max_k} possible for top K retrieved candidates if ranked ideally, '
             f'by sources (src_any=all sources, src_...=other sources separately): \n'
             + str(tabulate(metrics_all.to_pandas(), headers=metrics_all.columns, showindex=False)))

    file_out = f'{dir_out}/{get_submit_file_name("eval-retrieved-recall", args.tag)}.csv'
    metrics_all.write_csv(file_out, float_precision=4)
    log.info(f'Recalls saved to: {file_out}')

    df_retrieved = pl.scan_parquet(dir_retrieved).select(['session'] + srcs).collect()

    lst_summary = []
    for src in srcs:
        stats_summary = describe_numeric(df_retrieved.groupby('session').agg([pl.sum(src).alias('n')]).to_pandas()[['n']])
        lst_summary.append(pd.concat([pd.DataFrame({'source': [src]}), stats_summary.reset_index(drop=True)], axis=1, ignore_index=False))

    stats_summary = pd.concat(lst_summary).drop(columns=['count', 'std', 'count_nan', 'prc_nan'])
    file_out = f'{dir_out}/{get_submit_file_name("eval-retrieved-counts", args.tag)}.csv'
    log.info(f'Stats of number of aids per session, by source: \n{str(tabulate(stats_summary, headers=stats_summary.columns, showindex=False))}')
    stats_summary.to_csv(file_out, float_format='%.3f')
    log.info(f'Summary of candidates by sources saved to: {file_out}')



# ******************************************************************************
# Stats of number of aids per session, by source:
# source                          mean    min    1%    5%    10%    25%    50%    75%    90%    95%    99%    max
# ------------------------  ----------  -----  ----  ----  -----  -----  -----  -----  -----  -----  -----  -----
# src_any                   172.354        56    71    83     88    104    126    186    305    424    740   2322
# src_self                    3.06404       1     1     1      1      1      1      3      6     10     24    137
# src_click_to_click         20.474         0     0     2      9     10     10     21     41     63    133    553
# src_click_to_cart_or_buy   20.1775        0     0     0      3     10     10     20     42     66    139    569
# src_cart_to_cart            4.66161       0     0     0      0      0      0      0     20     29     80    657
# src_cart_to_buy             3.31507       0     0     0      0      0      0      0      7     22     63    517
# src_buy_to_buy              0.324088      0     0     0      0      0      0      0      0      0      9    266
# src_w2vec_all              42.7851        0     0     0     20     20     20     48     91    132    243    858
# src_w2vec_1_2              40.9391        0     0     0      0     20     20     41     92    133    242    861
# src_pop_cl50               56.9076       32    34    35     37     44     54     65     86     86     86     86

# The maximum recal@20 possible for top K retrieved candidates if ranked ideally,
# by sources (src_any=all sources, src_...=other sources separately):
# source                                   type      recall@20-top20    recall@20-top100    recall@20-top200    recall@20-topall
# ---------------------------------------  ------  -----------------  ------------------  ------------------  ------------------
# src_any                                  clicks        0.196203             0.5307             0.560093            0.569288
# src_any                                  carts         0.152458             0.424199           0.467714            0.50739
# src_any                                  orders        0.16003              0.481797           0.584761            0.713684
# src_any                                  total         0.161375             0.469408           0.54718             0.637356
# src_self                                 clicks        0.321673             0.322039           0.32204             0.32204
# src_self                                 carts         0.308162             0.311829           0.31184             0.31184
# src_self                                 orders        0.577951             0.598167           0.598196            0.598196
# src_self                                 total         0.471387             0.484653           0.484673            0.484673
# src_click_to_click                       clicks        0.435624             0.453825           0.454231            0.454271
# src_click_to_click                       carts         0.334155             0.376075           0.380299            0.380885
# src_click_to_click                       orders        0.413123             0.563529           0.582445            0.584574
# src_click_to_click                       total         0.391683             0.496322           0.50898             0.510437
# src_click_to_cart_or_buy                 clicks        0.365465             0.38157            0.381994            0.382036
# src_click_to_cart_or_buy                 carts         0.334858             0.377241           0.381934            0.382683
# src_click_to_cart_or_buy                 orders        0.418102             0.570776           0.59156             0.594301
# src_click_to_cart_or_buy                 total         0.387865             0.493795           0.507716            0.509589
# src_cart_to_cart                         clicks        0.0435721            0.0502754          0.0504181           0.0504348
# src_cart_to_cart                         carts         0.100796             0.128639           0.130476            0.130665
# src_cart_to_cart                         orders        0.257793             0.399807           0.416169            0.41815
# src_cart_to_cart                         total         0.189271             0.283503           0.293886            0.295133
# src_cart_to_buy                          clicks        0.0389317            0.0429231          0.0430019           0.0430077
# src_cart_to_buy                          carts         0.0869485            0.105198           0.106263            0.106337
# src_cart_to_buy                          orders        0.279137             0.399022           0.410817            0.411714
# src_cart_to_buy                          total         0.19746              0.275265           0.282669            0.28323
# src_buy_to_buy                           clicks        0.00355818           0.00381883         0.00382113          0.00382113
# src_buy_to_buy                           carts         0.00477473           0.0056456          0.00568269          0.00568269
# src_buy_to_buy                           orders        0.0191055            0.0255255          0.0257281           0.0257281
# src_buy_to_buy                           total         0.0132515            0.0173909          0.0175238           0.0175238
# src_w2vec_all                            clicks        0.334309             0.411028           0.413489            0.41388
# src_w2vec_all                            carts         0.271287             0.358414           0.370756            0.374832
# src_w2vec_all                            orders        0.302218             0.521088           0.576578            0.597562
# src_w2vec_all                            total         0.296148             0.46128            0.498523            0.512375
# src_w2vec_1_2                            clicks        0.26726              0.322112           0.323941            0.324257
# src_w2vec_1_2                            carts         0.251619             0.325367           0.336545            0.340315
# src_w2vec_1_2                            orders        0.29573              0.506434           0.561966            0.583413
# src_w2vec_1_2                            total         0.27965              0.433682           0.470537            0.484568
# src_pop_cl50                             clicks        0.0989431            0.109026           0.109026            0.109026
# src_pop_cl50                             carts         0.0775563            0.0842176          0.0842176           0.0842176
# src_pop_cl50                             orders        0.0794165            0.0839903          0.0839903           0.0839903
# src_pop_cl50                             total         0.0808111            0.086562           0.086562            0.086562
#