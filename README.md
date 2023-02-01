# OTTO – Multi-Objective Recommender System

## Description of the competition
> "The goal of this competition is to predict e-commerce clicks, cart additions, and orders. 
You'll build a multi-objective recommender system based on previous events in a user session."
* Kaggle: OTTO – Multi-Objective Recommender System]: https://www.kaggle.com/competitions/otto-recommender-system  
* Data documentation from provider: https://github.com/otto-de/recsys-dataset

## What's in the data?
* **12.9M** real-world anonymized user sessions ()
* **220M** events, consisting of clicks, carts and orders
* **1.8M** unique articles in the catalogue

### Number of items per session:

| action type | mean | std  |   min |   25% |   50% |   95% |   98% |   99% |   max |
|:------------|------|------|-------|-------|-------|-------|-------|-------|-------|
| clicks      | 3.41 | 6.35 |     1 |     1 |     2 |    12 |    20 |    28 |   465 |
| carts       | 2.22 | 2.63 |     1 |     1 |     1 |     7 |    10 |    14 |    53 |
| orders      | 1.85 | 1.97 |     1 |     1 |     1 |     5 |     8 |    10 |    31 |

## Solution
**High level**:  
Corpus of 1.8M articles > Retrieve ~56  candidates > Rank > Submit top 20 items for clicks, carts and orders.

### Retrieval
* **Re-visit**: Add all previous items from the session to the pool of candidates.
* **Co-visit**: Add items frequently visited together. Add 20 most frequent items per each previous item.
  * click-to-click: clicked together (in the span of 12 hours)
  * click-to-cart-or-buy: clicked one item and then added to cart or ordered the other (in the span of 24 hours)
  * cart-to-cart: added to cart together (24h)
  * cart-to-buy: added to cart one item and then ordered the other (24h)
  * buy-to-buy: ordered together (24h)
* **Similar items**: Add similar items in terms of Word2Vec embeddings. Add at most 20 similar items per each previous item.
  * based on Word2Vec embeddings from sequence of all items in the session
  * based on Word2Vec embeddings from sequence of only cart & buy items in the session
* **Popular items in the same cluster of sessions**: Compute sessions embeddings based 
  on Word2Vec embeddings of items in the session (a weighted average by type and time). 
  Find clusters of sessions using KMeans (~50 clusters found). Add top 20 most popular 
  items from the same cluster.

Number of candidates retrieved per session:
```
      mean    min    1%    5%    10%    25%    50%    75%    90%    95%    99%    max
----------  -----  ----  ----  -----  -----  -----  -----  -----  -----  -----  -----
   172.354     56    71    83     88    104    126    186    305    424    740   2322
```

<details>
<summary>In more details by source...</summary>
<p>

```
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
```
</p>
</details>

The maximum recal@20 possible for top K retrieved candidates if ranked ideally.
```
type      recall@20-top20    recall@20-top100    recall@20-top200    recall@20-topall
------  -----------------  ------------------  ------------------  ------------------
clicks        0.196203             0.5307             0.560093            0.569288
carts         0.152458             0.424199           0.467714            0.50739
orders        0.16003              0.481797           0.584761            0.713684
total         0.161375             0.469408           0.54718             0.637356
```

<details>
<summary>In more details by source...</summary>
<p>
(src_any=all sources, src_...=other sources separately) 

```
source                                   type      recall@20-top20    recall@20-top100    recall@20-top200    recall@20-topall
---------------------------------------  ------  -----------------  ------------------  ------------------  ------------------
src_any                                  clicks        0.196203             0.5307             0.560093            0.569288
src_any                                  carts         0.152458             0.424199           0.467714            0.50739
src_any                                  orders        0.16003              0.481797           0.584761            0.713684
src_any                                  total         0.161375             0.469408           0.54718             0.637356
src_self                                 clicks        0.321673             0.322039           0.32204             0.32204
src_self                                 carts         0.308162             0.311829           0.31184             0.31184
src_self                                 orders        0.577951             0.598167           0.598196            0.598196
src_self                                 total         0.471387             0.484653           0.484673            0.484673
src_click_to_click                       clicks        0.435624             0.453825           0.454231            0.454271
src_click_to_click                       carts         0.334155             0.376075           0.380299            0.380885
src_click_to_click                       orders        0.413123             0.563529           0.582445            0.584574
src_click_to_click                       total         0.391683             0.496322           0.50898             0.510437
src_click_to_cart_or_buy                 clicks        0.365465             0.38157            0.381994            0.382036
src_click_to_cart_or_buy                 carts         0.334858             0.377241           0.381934            0.382683
src_click_to_cart_or_buy                 orders        0.418102             0.570776           0.59156             0.594301
src_click_to_cart_or_buy                 total         0.387865             0.493795           0.507716            0.509589
src_cart_to_cart                         clicks        0.0435721            0.0502754          0.0504181           0.0504348
src_cart_to_cart                         carts         0.100796             0.128639           0.130476            0.130665
src_cart_to_cart                         orders        0.257793             0.399807           0.416169            0.41815
src_cart_to_cart                         total         0.189271             0.283503           0.293886            0.295133
src_cart_to_buy                          clicks        0.0389317            0.0429231          0.0430019           0.0430077
src_cart_to_buy                          carts         0.0869485            0.105198           0.106263            0.106337
src_cart_to_buy                          orders        0.279137             0.399022           0.410817            0.411714
src_cart_to_buy                          total         0.19746              0.275265           0.282669            0.28323
src_buy_to_buy                           clicks        0.00355818           0.00381883         0.00382113          0.00382113
src_buy_to_buy                           carts         0.00477473           0.0056456          0.00568269          0.00568269
src_buy_to_buy                           orders        0.0191055            0.0255255          0.0257281           0.0257281
src_buy_to_buy                           total         0.0132515            0.0173909          0.0175238           0.0175238
src_w2vec_all                            clicks        0.334309             0.411028           0.413489            0.41388
src_w2vec_all                            carts         0.271287             0.358414           0.370756            0.374832
src_w2vec_all                            orders        0.302218             0.521088           0.576578            0.597562
src_w2vec_all                            total         0.296148             0.46128            0.498523            0.512375
src_w2vec_1_2                            clicks        0.26726              0.322112           0.323941            0.324257
src_w2vec_1_2                            carts         0.251619             0.325367           0.336545            0.340315
src_w2vec_1_2                            orders        0.29573              0.506434           0.561966            0.583413
src_w2vec_1_2                            total         0.27965              0.433682           0.470537            0.484568
src_pop_cl50                             clicks        0.0989431            0.109026           0.109026            0.109026
src_pop_cl50                             carts         0.0775563            0.0842176          0.0842176           0.0842176
src_pop_cl50                             orders        0.0794165            0.0839903          0.0839903           0.0839903
src_pop_cl50                             total         0.0808111            0.086562           0.086562            0.086562
```
</p>
</details>


### Features Engineering
* Item within session: count by type, rank, relative position within session, time since last time clicked/carted/ordered
* Co-occurence counts by type, ranks by these counts (if more previous items paired with the same next item, aggregate, take last, take best)
* Word2Vec similarity in terms of euclidian distance and also the ranks
* Similarity between session embedding and item embedding: cosine, euclidian distance
* Popularity rank within the same session cluster

Overall, about 100 features were created. However, this proved to be insufficient to rank decently in the competition.

<details>
<summary>Feature importance from LightGBM model for orders...</summary>
<p>

```
                            feature  importance
          click_to_click_perc_pop       0.365
                            slf_n       0.154
                         src_self       0.127
             click_to_click_count       0.046
                    slf_rank_by_n       0.040
         click_to_click_count_rel       0.027
               src_click_to_click       0.026
                   dist_w2vec_all       0.026
                  cos_sim_ses_aid       0.024
             slf_rank_by_n_orders       0.019
              since_ts_aid_clicks       0.015
              click_to_click_rank       0.015
   click_to_cart_or_buy_count_pop       0.014
         click_to_click_count_pop       0.012
                     ts_order_aid       0.010
              slf_rank_by_n_carts       0.007
              slf_ts_order_clicks       0.007
           cart_to_cart_count_rel       0.007
              best_rank_w2vec_all       0.006
                     slf_since_ts       0.005
       click_to_cart_or_buy_count       0.004
                     since_ts_aid       0.004
              slf_since_ts_clicks       0.004
                eucl_dist_ses_aid       0.003
              rank_clicks_7d_cl50       0.003
```
</p>
</details>

<details>
<summary>Feature importance from LightGBM model for carts...</summary>
<p>

```
                            feature  importance
34       click_to_cart_or_buy_count       0.288
0                             slf_n       0.185
83                         src_self       0.157
31          click_to_click_perc_pop       0.068
4                     slf_rank_by_n       0.046
35   click_to_cart_or_buy_count_pop       0.029
55                   dist_w2vec_all       0.024
73                     slf_since_ts       0.017
101                 cos_sim_ses_aid       0.016
6              slf_rank_by_n_orders       0.013
74              slf_since_ts_clicks       0.011
70              since_ts_aid_clicks       0.011
20                      n_aid_carts       0.011
38   click_to_cart_or_buy_count_rel       0.010
22                     ts_order_aid       0.010
48            cart_to_buy_count_rel       0.010
42                cart_to_cart_rank       0.010
1                      slf_n_clicks       0.008
37        click_to_cart_or_buy_rank       0.008
33         click_to_click_count_rel       0.007
14                       n_uniq_aid       0.007
10              slf_ts_order_clicks       0.006
43           cart_to_cart_count_rel       0.006
69                     since_ts_aid       0.005
29             click_to_click_count       0.004
```
</p>
</details>

<details>
<summary>Feature importance (from LightGBM) for <b>orders</b>...</summary>
<p>

```
                            feature  importance
34       click_to_cart_or_buy_count       0.281
0                             slf_n       0.192
83                         src_self       0.176
48            cart_to_buy_count_rel       0.060
4                     slf_rank_by_n       0.049
2                       slf_n_carts       0.043
58                      n_w2vec_1_2       0.027
35   click_to_cart_or_buy_count_pop       0.022
74              slf_since_ts_clicks       0.017
6              slf_rank_by_n_orders       0.013
31          click_to_click_perc_pop       0.011
70              since_ts_aid_clicks       0.009
101                 cos_sim_ses_aid       0.008
1                      slf_n_clicks       0.007
38   click_to_cart_or_buy_count_rel       0.007
44                cart_to_buy_count       0.006
22                     ts_order_aid       0.006
13                 slf_left_in_cart       0.005
73                     slf_since_ts       0.005
42                cart_to_cart_rank       0.005
5               slf_rank_by_n_carts       0.004
37        click_to_cart_or_buy_rank       0.004
54                      n_w2vec_all       0.003
43           cart_to_cart_count_rel       0.003
59                   dist_w2vec_1_2       0.003
```
</p>
</details>


### Model
* Target: a retrieved item is marked with 1 if item was clicked, carted, ordered; and 0 if not.
* Removed sessions without positive samples. This decreased the volume of rows to 
  ~13% of the original dataset for clicks, ~3.5% for carts, and ~2.5% for orders.
* Negative downsampling was done to a ratio of 1:40 positive:negative samples, 
  but maximum 100 negative samples per session. The datasets had the following: 

| Type   | Avg positive / session | Avg negative / session | Total rows | Sessions |
|--------|------------------------|------------------------|------------|----------|
| Clicks | 1                      | 41                     | 40M        | 1M       |
| Carts  | 1.3                    | 50                     | 11M        | 220K     |
| Orders | 1.7                    | 57                     | 7.5M       | 130K     |

* Train 3 LightGBM(lambdarank) models for each type clicks/carts/orders.
* No parameters tuning was done (lack of time). Used the same parameters for all models.
 
  ```python
  PARAMS_LGBM = {
    'objective': 'lambdarank',
    'boosting_type': 'gbdt',  # 'gbdt', # 'dart',
    'metric': 'ndcg',
    'n_estimators': 150,
    'learning_rate': 0.25,  # use higher for orders ~0.50?, and lower for carts ~0.01?
    'max_depth': 4,
    'num_leaves': 15,
    'colsample_bytree': 0.25,  # aka feature_fraction
    'subsample': 0.50,  # aka bagging_fraction
    # 'bagging_freq': 1,
    'min_child_samples': 20,  # aka min_data_in_leaf  ? read github link with test
    'importance_type': 'gain',
    'seed': 42,
  }
  ```

## Pipeline
1. Create env from [requirements.txt](requirements.txt) or [conda.yml](conda.yml) and activate it. `cd` to project root.

2. Download data from kaggle to `data/full` folder.
   Rename files to `train_sessions.jsonl` and `test_session.jsonl`.

3. Split `train_sessions.jsonl` to train/test with [etl/split_to_train_test.sh](etl/split_to_train_test.sh). 
   Test to have 1 week. Splitted data located in `data/train-test` folder.
```shell
./split_train_test.sh
```

4. Optimize data format for better I/O. 
   Convert jsonl to parquet using [etl/jsonl_to_parquet.py](etl/jsonl_to_parquet.py)
```shell
python -m etl/jsonl_to_parquet.py --data_split_alias full
python -m etl/jsonl_to_parquet.py --data_split_alias train-test
```

5. Count co-occurrences for items with sessions with [model/count_co_events.py](model/count_co_events.py)
```shell
python -m model/count_co_events.py --data_split_alias full
python -m model/count_co_events.py --data_split_alias train-test
```

6. Run Word2Vec models specified in `config.W2VEC_MODELS` with [model/w2vec_aids.py](model/w2vec_aids.py) 
   and find top similar candidates for all items in the index using faiss.
```shell
python -m model/w2vec_aids.py
```

7. Find clusters for session with [model/kmeans_sessions.py](model/kmeans_sessions.py)
```shell
python -m model/kmeans_sessions.py --data_split_alias train-test --model_name word2vec-train-test-types-all-size-100-mincount-5-window-10
python -m model/kmeans_sessions.py --data_split_alias full --model_name word2vec-full-types-all-size-100-mincount-5-window-10
```

8. Count popularity ranks of aids within session clusters with [model/count_popularity.py](model/count_popularity.py)
```shell
python -m model/count_popularity.py --data_split_alias train-test
python -m model/count_popularity.py --data_split_alias full
```

9. Retrieve candidates and generate features with [model/retrieve.py](model/retrieve.py)
```shell
python -m model/retrieve.py --data_split_alias train-test --w2vec_model_all word2vec-train-test-types-all-size-100-mincount-5-window-10 --w2vec_model_1_2 word2vec-train-test-types-1-2-size-100-mincount-5-window-10
python -m model/retrieve.py --data_split_alias full --w2vec_model_all word2vec-full-types-all-size-100-mincount-5-window-10 --w2vec_model_1_2 word2vec-full-types-1-2-size-100-mincount-5-window-10
```

10. Evaluate the retrieved candidates with [model/eval_retrieved.py](model/eval_retrieved.py)
```shell
python -m model/eval_retrieved.py
```

11. Downsample retrieved data set with [model/downsample_retrieved.py](model/downsample_retrieved.py)
```shell 
python -m model/downsample_retrieved.py
```

12. Train LightGBM models with [model/train_lgbm_rankers.py](model/train_lgbm_rankers.py)
```shell
python -m model/train_lgbm_rankers.py
```

13. Rank retrieved candidates with [model/rank.py](model/rank.py)
```shell
python -m model/rank.py --data_split_alias train-test
python -m model/rank.py --data_split_alias full
```

14. Make the submission file with [model/submit.py](model/submit.py)
```shell
python -m model/submit.py --data_split_alias train-test
python -m model/submit.py --data_split_alias full
```

15. Evaluate the submission with [model/eval_submission.py](model/eval_submission.py)
```shell
python -m model/eval_submission.py
```


## References
### Data optimization
* [[Howto] Full dataset as parquet/csv files](https://www.kaggle.com/code/radek1/howto-full-dataset-as-parquet-csv-files)
* [OTTO - Fast DataFrame Loading in Parquet format](https://www.kaggle.com/code/columbia2131/otto-fast-dataframe-loading-in-parquet-format)

### EDA
* [One Month Left - Here is what you need to know!](https://www.kaggle.com/competitions/otto-recommender-system/discussion/374229)
* [OTTO - Getting Started (EDA + Baseline)](https://www.kaggle.com/code/edwardcrookenden/otto-getting-started-eda-baseline)

### Model
* https://www.kaggle.com/code/cdeotte/candidate-rerank-model-lb-0-575
* https://www.kaggle.com/competitions/otto-recommender-system/discussion/368560
* [Word2Vec How-to [training and submission]](https://www.kaggle.com/code/radek1/word2vec-how-to-training-and-submission)
* https://www.kaggle.com/code/vslaykovsky/co-visitation-matrix
* https://www.kaggle.com/code/cdeotte/test-data-leak-lb-boost
* https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/

### General
* [Top 200 : Empirical results/remarks about training the ranker.](https://www.kaggle.com/competitions/otto-recommender-system/discussion/381469)

### Word2Vec
* https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html
* https://radimrehurek.com/gensim/models/word2vec.html
* https://jalammar.github.io/illustrated-word2vec/
* https://www.kaggle.com/code/radek1/word2vec-how-to-training-and-submission
* https://www.kaggle.com/competitions/otto-recommender-system/discussion/368384

### Faiss (for fast similarity search)
* https://www.pinecone.io/learn/faiss-tutorial/
* https://davidefiocco.github.io/nearest-neighbor-search-with-faiss/
* https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/

### Annoy (for fast similarity search) - not used
* https://github.com/spotify/annoy
* https://radimrehurek.com/gensim/auto_examples/tutorials/run_annoy.html

### K-means clustering (with Dask)
* https://realpython.com/k-means-clustering-python/
* https://dask.pydata.org/en/latest/array-creation.html

### LightGBM ranker
* https://medium.datadriveninvestor.com/a-practical-guide-to-lambdamart-in-lightgbm-f16a57864f6
* https://github.com/jameslamb/lightgbm-dask-testing/blob/main/notebooks/demo.ipynb
* https://www.kaggle.com/code/radek1/polars-proof-of-concept-lgbm-ranker
* https://lightgbm.readthedocs.io/en/latest/Parameters.html
* https://lightgbm.readthedocs.io/en/latest/Parallel-Learning-Guide.html#how-distributed-lightgbm-works
* https://github.com/microsoft/LightGBM/blob/477cbf373ea2138a186568ac88ef221ac74c7c71/tests/python_package_test/test_dask.py#L480
* https://www.kaggle.com/competitions/otto-recommender-system/discussion/381469
* https://www.kaggle.com/code/greenwolf/lightgbm-fast-recall-20/
* https://www.kaggle.com/competitions/otto-recommender-system/discussion/380732
