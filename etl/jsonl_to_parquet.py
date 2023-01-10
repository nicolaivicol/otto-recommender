import os
import numpy as np
import pandas as pd
import math
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Union

id2type = ['clicks', 'carts', 'orders']
type2id = {'clicks': 0, 'carts': 1, 'orders': 2}
# df['type'].map(type2id)  # df['type'].map(lambda i: id2type[i])


def get_number_of_lines(file_path) -> int:
    with open(file_path, 'r') as fp:
        num_lines = sum(1 for line in fp)
    return num_lines


def optimize_df_from_jsonl(df: pd.DataFrame) -> pd.DataFrame:
    df['session'] = df['session'].astype(np.int32)
    df['aid'] = df['aid'].astype(np.int32)
    df['type'] = df['type'].map(type2id).astype(np.int8)  # map string to int
    if 'ts' in df.columns:
        df['ts'] = (df['ts'] / 1000).astype(np.int32)  # milliseconds to seconds
    return df


def collect_events_to_columns(chunk) -> Dict[str, List[Union[str, int]]]:
    columns = {'session': [], 'aid': [], 'ts': [], 'type': []}

    for session, events in zip(chunk['session'].tolist(), chunk['events'].tolist()):
        for event in events:
            columns['session'].append(session)
            columns['aid'].append(event['aid'])
            columns['ts'].append(event['ts'])
            columns['type'].append(event['type'])

    return columns


def collect_labels_to_columns(chunk: pd.DataFrame) -> Dict[str, List[Union[str, int]]]:
    columns = {'session': [], 'type': [], 'aid': []}

    for session, labels in zip(chunk['session'].tolist(), chunk['labels'].tolist()):
        for type, aid in labels.items():
            if not isinstance(aid, list):
                aid = [aid]
            columns['session'].extend([session] * len(aid))
            columns['type'].extend([type] * len(aid))
            columns['aid'].extend(aid)

    return columns


def transform_jsonl_to_parquet(file_jsonl, out_dir, type_data='sessions', chunksize=100000):
    name_folder_parquets = Path(file_jsonl).stem
    dir_parquets = f'{out_dir}/{name_folder_parquets}'
    os.makedirs(dir_parquets, exist_ok=True)

    n_lines = get_number_of_lines(file_jsonl)
    n_chunks = math.ceil(float(n_lines)/chunksize)

    df_chunks = pd.read_json(file_jsonl, lines=True, chunksize=chunksize)

    for i, df_chunk in enumerate(tqdm(df_chunks, total=n_chunks, unit='chunk')):
        if type_data=='sessions':
            columns = collect_events_to_columns(df_chunk)
        elif type_data == 'labels':
            columns = collect_labels_to_columns(df_chunk)
        else:
            raise ValueError(f'type_data={type_data} not recognized, must be \'sessions\' or \'labels\'')

        df = pd.DataFrame(columns)
        df = optimize_df_from_jsonl(df)

        # save DataFrame to parquet
        n_digits = len(str(n_lines))
        start = str(i * chunksize).zfill(n_digits)
        end = str(i * chunksize + chunksize).zfill(n_digits)
        df.to_parquet(f'{dir_parquets}/{start}_{end}.parquet', index=False)


if __name__ == '__main__':
    dir_jsonl = '../data/train-test'
    dir_parquet = '../data/train-test-parquet'

    transform_jsonl_to_parquet(f'{dir_jsonl}/test_labels.jsonl', dir_parquet, 'labels')
    transform_jsonl_to_parquet(f'{dir_jsonl}/test_sessions.jsonl', dir_parquet)
    transform_jsonl_to_parquet(f'{dir_jsonl}/test_sessions_full.jsonl', dir_parquet)
    transform_jsonl_to_parquet(f'{dir_jsonl}/train_sessions.jsonl', dir_parquet)  # takes ~15min
