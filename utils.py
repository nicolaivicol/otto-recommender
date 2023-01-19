import os
import shutil
import math
import numpy as np
import pandas as pd
import polars as pl
from typing import List
from IPython.display import display, HTML
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings


HEIGHT_PLOT = 650


def describe_numeric(df: pd.DataFrame, cols_num: List[str]=None, percentiles: List[float]=None) -> pd.DataFrame:
    """
    Describe numeric columns
    :param df: pandas data frame
    :param cols_num: numeric columns to describe, by default: identified automatically
    :param percentiles: percentiles to compute, default: [0.05, 0.25, 0.50, 0.75, 0.95]
    :return: pandas df with stats
    """
    if cols_num is None:
        cols_num = list(df.head(1).select_dtypes(include=['number']).columns)
    if percentiles is None:
        percentiles = [0.05, 0.25, 0.50, 0.75, 0.95]
    if len(cols_num) == 0:
        return None
    d_describe = df[cols_num].describe(percentiles=percentiles).T
    d_describe['count_nan'] = df.isnull().sum()
    d_describe['prc_nan'] = 1 - d_describe['count'] / float(df.shape[0])
    return d_describe


def set_display_options():
    """
    Set display options for numbers, table width, etc.
    :return: None
    """
    pd.set_option('plotting.backend', 'plotly')
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', 50)
    pd.set_option('display.width', 2000)
    pd.set_option('display.max_colwidth', 150)
    pd.set_option('max_colwidth', 150)
    pd.set_option('display.precision', 2)
    pd.set_option('display.chop_threshold', 1e-6)
    # pd.set_option('expand_frame_repr', True)
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    warnings.simplefilter('ignore')
    # display(HTML("<style>.container { width:80% !important; }</style>"))


# def remove_folder(dir):
#     if os.path.exists(dir):
#         try:
#             shutil.rmtree(dir)
#         except OSError as e:
#             warnings.warn("Error: %s - %s." % (e.filename, e.strerror))


def compute_recall_at_k(session, target, pred_score, is_retrieved=None, k=20):
    if is_retrieved is None:
        is_retrieved = np.repeat(1, len(session))

    df_tmp = pl.DataFrame({'session': session, 'target': target, 'pred_score': pred_score, 'is_retrieved': is_retrieved})
    df_tmp = df_tmp \
        .select([
            pl.all(),
            (pl.col('pred_score').rank('ordinal', reverse=True).over('session').cast(pl.Int16).alias('pred_rank')),
        ]) \
        .sort(['session', 'pred_rank'])

    df_tmp = df_tmp \
        .groupby('session') \
        .agg([(pl.col('target') * pl.col('is_retrieved') * (pl.col('pred_rank') <= k)).sum().alias('hit'),
              pl.sum('target').clip_max(k)]) \
        .sum() \
        .select(pl.col('hit') / pl.col('target'))

    return df_tmp[0, 0]

