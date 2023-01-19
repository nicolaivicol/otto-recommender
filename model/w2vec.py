import os.path
import time
import logging
from tqdm import tqdm
from typing import List, Union, Dict
import json
import random
import numpy as np
import polars as pl
from gensim.models import Word2Vec
from annoy import AnnoyIndex
import faiss

import config
from utils import set_display_options

set_display_options()
log = logging.getLogger(os.path.basename(__file__))

# references:
# https://www.kaggle.com/code/radek1/word2vec-how-to-training-and-submission
# https://www.kaggle.com/competitions/otto-recommender-system/discussion/368384
# https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html
# https://radimrehurek.com/gensim/models/word2vec.html
# https://radimrehurek.com/gensim/auto_examples/tutorials/run_annoy.html
# https://www.pinecone.io/learn/faiss-tutorial/
# https://davidefiocco.github.io/nearest-neighbor-search-with-faiss/
# https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/
# https://github.com/spotify/annoy
# https://jalammar.github.io/illustrated-word2vec/


# number of AIDs per session in train/test:
#               count   mean    std   min    5%   10%   25%   50%    95%     98%     99%     max
# train: 10584517.000 15.442 29.418 2.000 2.000 2.000 3.000 6.000 62.000 108.000 152.000 498.000
# test:   1783737.000  4.250  8.388 1.000 1.000 1.000 1.000 2.000 15.000  27.000  38.000 465.000
#
# 782842 unique AIDs with 'clicks'
# 50871 unique AIDs with 'orders'

# word2vec embeddings (all types, size=100, mincount=5, window=10),
# retrieve top 20 similar with annoy/w2vec/faiss/faiss_ivff
# - compare to top click-to-click
#          aid  time_ann  time_w2vec  time_faiss  time_faiss_ivff  co-countXannoy  co-countXw2vec  co-countXfaiss  co-countXfaiss_ivff  annoyXw2vec  faissXw2vec  faiss_ivffXw2vec  faissXfaiss_ivff
# 0 928838.322     0.064       0.077       0.085            0.006           0.237           0.126           0.245                0.242        0.204        0.256             0.247             0.973
# - compare to top buy-to-buy
#          aid  time_ann  time_w2vec  time_faiss  time_faiss_ivff  co-countXannoy  co-countXw2vec  co-countXfaiss  co-countXfaiss_ivff  annoyXw2vec  faissXw2vec  faiss_ivffXw2vec  faissXfaiss_ivff
# 0 926372.715     0.030       0.079       0.074            0.004           0.185           0.263           0.199                0.197        0.379        0.420             0.411             0.980


def load_sessions_lazy(file: str, types: List[int] = None) -> pl.LazyFrame:
    """ Load a lazy data frame without collection/execution """
    cols_agg = [pl.count().cast(pl.UInt16).alias('n_aid'), pl.col('aid').alias('sentence')]
    if types is None:
        return pl.scan_parquet(file).groupby('session').agg(cols_agg)
    else:
        return pl.scan_parquet(file).filter(pl.col('type').is_in(types)).groupby('session').agg(cols_agg)


def load_sessions_as_sentences(files_or_dirs: Union[str, List[str]], types: List[str] = None) -> pl.DataFrame:
    """
    Load sessions as sentences, where AIDs are the words
    :param files_or_dirs: files/dirs with parquet files with sessions
    :param types: which types to load, by default all types loaded, i.e. [0, 1, 2]
    :return: pl.DataFrame
    """
    if not isinstance(files_or_dirs, list):
        files_or_dirs = [files_or_dirs]

    df_sentences = pl.concat([load_sessions_lazy(f, types) for f in files_or_dirs]).collect()
    log.debug(f'loaded {df_sentences.shape[0]} sessions, from {",".join(files_or_dirs)}')
    return df_sentences


def get_model_file(model_name):
    return f'{config.DIR_ARTIFACTS}/word2vec/{model_name}.model'


def load_w2vec_model(model_name) -> Word2Vec:
    model_file = get_model_file(model_name)

    if config.W2VEC_USE_CACHE and os.path.exists(model_file):
        log.debug(f'loading Word2Vec model from cache file {model_file}')
        return Word2Vec.load(model_file)
    else:
        return train_w2vec_model(model_name)


def train_w2vec_model(model_name) -> Word2Vec:
    tic = time.time()
    model_config = config.W2VEC_MODELS[model_name]
    log.debug(f'training Word2Vec with name=\'{model_name}\' '
                 f'and config: {json.dumps(model_config, indent=2)}')
    df_sentences = load_sessions_as_sentences(model_config['dir_sessions'], types=model_config.get('types', None))
    w2vec_model = Word2Vec(sentences=df_sentences['sentence'].to_list(), workers=16, **model_config['params'])

    log.debug(f'time elapsed: {time.strftime("%Hh %Mmin %Ssec", time.gmtime(time.time() - tic))}')

    w2vec_model.save(get_model_file(model_name))
    # w2vec_model.wv.save()
    log.debug(f"model saved to: {get_model_file(model_name)}")

    return w2vec_model


def load_annoy_index(words: List[int], embeddings: List[List[float]], model_name: str):
    # not used
    vector_size = config.W2VEC_MODELS[model_name]['params'].get('vector_size', 100)
    n_trees = config.W2VEC_MODELS[model_name]['n_trees']
    file_index = f'{get_model_file(model_name)}.index-ntrees-{n_trees}.ann'
    word2idx = {word: i for i, word in enumerate(words)}
    index_ann = AnnoyIndex(vector_size, 'euclidean')

    try:
        index_ann.load(file_index)
        log.debug(f'loaded index from: {file_index}')
    except Exception as e:
        log.warning(f'start building the index, could not load index from file: {file_index}, error: {str(e)}')

        for aid, idx in word2idx.items():
            index_ann.add_item(idx, embeddings[idx])

        index_ann.build(n_trees=config.W2VEC_MODELS[model_name]['n_trees'])  # eta ~10min for trees=100
        index_ann.save(file_index)
        log.debug(f'saved index to: {file_index}')

    return index_ann


def load_index_faiss_ivff(embeddings: List[List[float]], model_name: str):
    # https://www.pinecone.io/learn/faiss-tutorial/
    nlist = config.W2VEC_MODELS[model_name].get('nlist', 100)  # how many cells
    nprobe = config.W2VEC_MODELS[model_name].get('nprobe', 3)  # how many closest cells to search
    vector_size = len(embeddings[0])
    quantizer = faiss.IndexFlatL2(vector_size)
    index = faiss.IndexIVFFlat(quantizer, vector_size, nlist, faiss.METRIC_L2)
    index.train(np.array(embeddings))
    index.add(np.array(embeddings))
    index.nprobe = nprobe
    assert index.ntotal == len(embeddings)  # number of embeddings indexed
    assert index.is_trained is True
    return index


def _return_miscellaneous_for_get_top_k_similar_faiss(words_q, return_itself):
    if return_itself:
        nereast_words = [[word] for word in words_q]
        dists = [[0.0] for _ in words_q]
    else:
        nereast_words = [[] for _ in words_q]
        dists = [[] for _ in words_q]
    res = {'word': words_q, 'nearest_words': nereast_words, 'dist_w2vec': dists}
    res = pl.DataFrame(res).explode(['nearest_word', 'dist_w2vec'])
    return res


def get_top_k_similar_faiss(
        words_q: List[int],
        words: List[int],
        map_word_embedding: Dict[int, List[int]],
        index_faiss_ivff: faiss.IndexIVFFlat,
        k: int = 20,
        return_itself=True,
):
    """
    Get top-k nearest neighbours for a list of Aids
    :param words_q: for which words to find neighbours
    :param words: corpus
    :param map_word_embedding: map word to embbeding
    :param index_faiss_ivff: index to use
    :param k: number of neighbours to retrieve
    :param return_itself: return the aid itself as neighbour
    :return:
    """
    if len(words_q) > 100_000:
        # (!) Speed deteriorates for words (AIDs) lower in the embeddings,
        # because the embedding vectors have smaller values, and it is harder
        # to find neighbours for them. These words are less frequent and/or less
        # consistent, without "clear" embeddings, around the 0-origin in the hyperplane.
        # Their neighbours are also of low quality and are likely poor recommendations.
        aids_per_sec_map = [1400, 1300, 1250, 1070, 930, 830, 705, 580, 485, 430, 380]
        aids_per_sec = aids_per_sec_map[min(int(len(words_q) / 100_000), len(aids_per_sec_map))]
        eta = len(words_q) / aids_per_sec
        msg = (f'ETA is {time.strftime("%Hh %Mmin %Ssec", time.gmtime(eta))} '
               f'for get_top_k_similar_faiss() on {len(words_q)} words')
        log.warning(msg)

    words_q_embeddings = [map_word_embedding.get(word) for word in words_q]
    which_word_has_embedding = [i for i, embedding in enumerate(words_q_embeddings) if embedding is not None]

    if len(which_word_has_embedding) == 0:
        return _return_miscellaneous_for_get_top_k_similar_faiss(words_q, return_itself)

    words_q_found = [words_q[i] for i in which_word_has_embedding]
    words_q_embeddings_found = [words_q_embeddings[i] for i in which_word_has_embedding]
    dists, idxs = index_faiss_ivff.search(np.array(words_q_embeddings_found), k=k)  # list of lists
    nereast_words = [[words[i] for i in idxs_nereast_words] for idxs_nereast_words in idxs]  # replace indices with AIDs ("words")

    res = pl.DataFrame({'aid': words_q_found, 'aid_next': nereast_words, 'dist_w2vec': dists}). \
        explode(['aid_next', 'dist_w2vec']). \
        select([pl.all().cast(pl.Int32),
                (pl.col('dist_w2vec').rank('ordinal').over('aid').
                 clip_max(127).cast(pl.Int8).alias('rank_w2vec'))])

    return res


def retrieve_w2vec_knns_via_faiss_index(model_name: str, k: int = None, first_n_aids: int = None) -> pl.DataFrame:
    """
    Retrieve top-k neighbours based on w2vec embeddings via faiss index
    :param model_name: model name (alias) as in config, e.g. 'word2vec-train-test-types-all-size-100-mincount-5-window-10'
    :param k:  number of neighbours to retrieve
    :param first_n_aids: for how many AIDs (words) to find neighbours (output df will have first_n_aids*k rows)
    :return:
    """

    if k is None:
        k = config.W2VEC_MODELS[model_name].get('k', 20)

    if first_n_aids is None:
        first_n_aids = config.W2VEC_MODELS[model_name].get('first_n_aids', 600_000)

    file_nns = f'{get_model_file(model_name)}.top-{k}-nns-{first_n_aids}-aids.parquet'

    if config.W2VEC_USE_CACHE and os.path.exists(file_nns):
        log.info('loading KNNs from cache')
        return pl.read_parquet(file_nns)

    w2vec_model = load_w2vec_model(model_name)
    words = w2vec_model.wv.index_to_key  # words sorted by "importance"
    embeddings = w2vec_model.wv.vectors
    word2idx = {word: i for i, word in enumerate(words)}  # map word to index
    map_word_embedding = {word: embeddings[word2idx[word]] for word in words}  # map word (aid) to its embedding
    index_faiss_ivff = load_index_faiss_ivff(embeddings, model_name)
    df_nns = get_top_k_similar_faiss(words[:first_n_aids], words, map_word_embedding, index_faiss_ivff, k)
    df_nns.write_parquet(file_nns)

    return df_nns


if __name__ == '__main__':
    # Train all models from config
    os.makedirs(f'{config.DIR_ARTIFACTS}/word2vec', exist_ok=True)
    for model_name in config.W2VEC_MODELS.keys():
        log.info(f'train word2vec model \'{model_name}\' to generate embeddings, '
                    f'then retrieve top-k nearest AIDs after indexing the embeddings.')
        log.debug(f'config: \n {json.dumps(config.W2VEC_MODELS[model_name], indent=2)}')
        df_knns = retrieve_w2vec_knns_via_faiss_index(model_name)
        n_candidates, n_aids = df_knns.shape[0], len(df_knns['aid'].unique()),
        log.info(f'{n_candidates:,} candidates retrieved for {n_aids:,} unique AIDs')




# ******************************************************************************
# df_click_to_click = pl.read_parquet('../data/train-test-counts-co-event/count_click_to_click.parquet')
# aids = list(df_click_to_click['aid'].unique())
#
#
# def check_random_aid():
#     aid_q = random.choice(aids)
#     # aid_q = 1741861
#     tic = time.time()
#     nns_idx, nns_dist = index_ann.get_nns_by_item(word2idx[aid_q], 20, include_distances=True)
#     nns_aid = [w2vec_model.wv.index_to_key[i] for i in nns_idx]
#     time_ann = time.time() - tic
#
#     df_nns_ann = pl.DataFrame({'aid_next': nns_aid, 'dist_ann': nns_dist}).\
#         with_columns(pl.col('aid_next').cast(pl.Int32, strict=False))
#
#     tic = time.time()
#     nns_w2vsim = w2vec_model.wv.most_similar(aid_q, topn=20)
#     time_w2vsim = time.time() - tic
#
#     df_nns_w2vec = pl.DataFrame(nns_w2vsim, columns=['aid_next', 'sim_w2vec']).\
#         with_columns(pl.col('aid_next').cast(pl.Int32, strict=False))
#
#     # tic = time.time()
#     # aid_q_embedding = w2vec_model.wv.vectors[aid2idx[aid_q]]
#     # dists, idxs = index_faiss.search(np.array([aid_q_embedding]), k=20)
#     # knns_faiss = [w2vec_model.wv.index_to_key[i] for i in idxs[0]]
#     # # knns_faiss = random.sample(aids, 20)
#     # dists_faiss = list(dists[0])
#     # time_faiss = time.time() - tic
#     # df_nns_faiss = pl.DataFrame({'aid_next': knns_faiss, 'dist_faiss': dists_faiss}).\
#     #     with_columns(pl.col('aid_next').cast(pl.Int32, strict=False))
#
#     tic = time.time()
#     aid_q_embedding = w2vec_model.wv.vectors[word2idx[aid_q]]
#     dists, idxs = index_faiss_ivff.search(np.array([aid_q_embedding]), k=20)
#     knns_faiss_ivff = [w2vec_model.wv.index_to_key[i] for i in idxs[0]]
#     dists_faiss_ivff = list(dists[0])
#     time_faiss_ivff = time.time() - tic
#
#     df_nns_faiss_ivff = pl.DataFrame({'aid_next': knns_faiss_ivff, 'dist_faiss_ivff': dists_faiss_ivff}).\
#         with_columns(pl.col('aid_next').cast(pl.Int32, strict=False))
#
#     df_nns = df_nns_ann\
#         .join(df_nns_w2vec, how='outer', on='aid_next')\
#         .join(df_nns_faiss_ivff, how='outer', on='aid_next')
#
#     # .join(df_nns_faiss, how='outer', on='aid_next') \
#
#     df_tmp = df_click_to_click\
#         .filter(pl.col('aid') == aid_q)\
#         .select([pl.all(), pl.col('aid_next').cumcount().alias('rank')])\
#         .join(df_nns, on='aid_next', how='outer')\
#         .to_pandas()
#     #         .filter((~pl.col('dist_ann').is_null())
#     #                 | (~pl.col('sim_w2vec').is_null())
#     #                 | (~pl.col('dist_faiss').is_null())
#     #                 | (~pl.col('dist_faiss_ivff').is_null()))\
#
#     r = {'aid': aid_q,
#          'time_ann': time_ann,
#          'time_w2vec': time_w2vsim,
#          # 'time_faiss': time_faiss,
#          'time_faiss_ivff': time_faiss_ivff,
#          'co-countXannoy': sum((df_tmp['dist_ann'] >= 0) * (~np.isnan(df_tmp['aid']))) / min(20, sum(~np.isnan(df_tmp['aid']))),
#          'co-countXw2vec': sum((df_tmp['sim_w2vec'] >= 0) * (~np.isnan(df_tmp['aid']))) / min(20, sum(~np.isnan(df_tmp['aid']))),
#          #'co-countXfaiss': sum((df_tmp['dist_faiss'] >= 0) * (~np.isnan(df_tmp['aid']))) / min(20, sum(~np.isnan(df_tmp['aid']))),
#          'co-countXfaiss_ivff': sum((df_tmp['dist_faiss_ivff'] >= 0) * (~np.isnan(df_tmp['aid']))) / min(20, sum(~np.isnan(df_tmp['aid']))),
#          'annoyXw2vec': sum((df_tmp['sim_w2vec'] >= 0) * (df_tmp['dist_ann'] >= 0)) / min(20, sum((df_tmp['sim_w2vec'] >= 0))),
#          #'faissXw2vec': sum((df_tmp['sim_w2vec'] >= 0) * (df_tmp['dist_faiss'] >= 0)) / min(20, sum((df_tmp['sim_w2vec'] >= 0))),
#          'faiss_ivffXw2vec': sum((df_tmp['sim_w2vec'] >= 0) * (df_tmp['dist_faiss_ivff'] >= 0)) / min(20, sum((df_tmp['sim_w2vec'] >= 0))),
#          #'faissXfaiss_ivff': sum((df_tmp['dist_faiss'] >= 0) * (df_tmp['dist_faiss_ivff'] >= 0)) / sum((df_tmp['dist_faiss'] >= 0)),
#          }
#
#     print(r)
#     # print(df_tmp)
#     return r
#
#
# lst_r = []
# for _ in range(200):
#     try:
#         lst_r.append(check_random_aid())
#     except Exception as e:
#         print(e)
#
# df_tmp = pl.DataFrame(lst_r)
# print(df_tmp.mean().to_pandas())
# df_tmp.write_csv(f'{config.DIR_ARTIFACTS}/stats_w2vec_x_co_click-{model_name}.csv')
#
# print('done')
#


