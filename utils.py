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
    :param percentiles: percentiles to compute, default: [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    :return: pandas df with stats
    """
    if cols_num is None:
        cols_num = list(df.head(1).select_dtypes(include=['number']).columns)
    if percentiles is None:
        percentiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
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


def get_last_commit_hash():
    try:
        import subprocess
        result = subprocess.check_output(['git', 'log', '-1', '--pretty=format:"%H"'])
        return result.decode('utf-8').replace('"', '')[:8]
    except Exception as e:
        return None


def get_timestamp():
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d%H%M%S")


def get_submit_file_name(prefix='submission', tag=None):
    tag = '' if tag is None else f'-{tag}'
    commit_hash = '' if get_last_commit_hash() is None else f'-{get_last_commit_hash()}'
    timestamp = f'-{get_timestamp()}'
    return f'{prefix}{timestamp}{tag}{commit_hash}'
