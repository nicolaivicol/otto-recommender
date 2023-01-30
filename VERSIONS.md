## Ver 1.0.0

### Train-test split
Split train sessions into train and test. Test sessions span over 7 days after 
the train sessions, like Kaggle's test.

### Retrieval
* Self-join all aids
* Add aid from co-counts:
  * _`click_to_click`_: 12h (before and after); top 10 (co-aids for each aid)
  * _`click_to_cart_or_buy`_: 24h; top 10
  * _`cart_to_cart`_: 24h; top 20
  * _`cart_to_buy`_: 24h; top 20
  * _`buy_to_buy`_: 24h; top 20   
  Pair with last 30 clicks only, pair with all carts and orders.
* Add top 20 similar aids based on word2vec trained on _all actions_ as sentence.
* Add top 20 similar aids based on word2vec trained on _carts and orders_ actions as sentence.

```
# Maximum recal@20 possible for top K candidates:
# ┌────────┬─────────────────┬──────────────────┬──────────────────┬──────────────────┐
# │ type   ┆ recall@20-top20 ┆ recall@20-top100 ┆ recall@20-top200 ┆ recall@20-topall │
# ╞════════╪═════════════════╪══════════════════╪══════════════════╪══════════════════╡
# │ clicks ┆ 0.174403        ┆ 0.47164          ┆ 0.522317         ┆ 0.550825         │
# │ carts  ┆ 0.10698         ┆ 0.340022         ┆ 0.417022         ┆ 0.494018         │
# │ orders ┆ 0.111423        ┆ 0.386607         ┆ 0.516885         ┆ 0.704218         │
# │ total  ┆ 0.116388        ┆ 0.381135         ┆ 0.487469         ┆ 0.625819         │
# └────────┴─────────────────┴──────────────────┴──────────────────┴──────────────────┘
```


### Ranking
Train 3 separate LightGBM rankers for clicks, carts and orders.  
Trained only on first 500K test session (due to memory limits).

### Results
Recall@20 on local test sessions: **0.5620** 
  * clicks: 0.4925 
  * carts: 0.4074
  * orders: 0.6508

Recall@20 on LB: **0.565** (without re-counting co-events)

## Ver 1.1.0
Adding more candidates from co-counts

## Ver 1.2.0
Adding candidates from clusters of sessions 
(KNN of 50 clusters, embedding based on word2vec embeddings of actions in the session).   

```
# Maximum recal@20 possible for top K candidates:
source,type,recall@20-top20,recall@20-top100,recall@20-top200,recall@20-topall
src_any,clicks,0.1593,0.4725,0.5396,0.5720
src_any,carts,0.0994,0.3371,0.4266,0.5091
src_any,orders,0.1057,0.3809,0.5236,0.7146
src_any,total,0.1092,0.3769,0.4961,0.6387
```

## Ver 1.3.0
Adding candidates from clusters of sessions, only from KNN of 50 clusters.

The maximum recal@20 possible for top K retrieved candidates if ranked ideally, by sources (src_any=all sources, src_...=other sources separately):
```
source      type      recall@20-top20    recall@20-top100    recall@20-top200    recall@20-topall
---------   ------  -----------------  ------------------  ------------------  ------------------
src_any     clicks        0.196203             0.5307             0.560093            0.569288
src_any     carts         0.152458             0.424199           0.467714            0.50739
src_any     orders        0.16003              0.481797           0.584761            0.713684
src_any     total         0.161375             0.469408           0.54718             0.637356
```

Stats of number of aids per session, by source: 
```
source                          mean    min    1%    5%    10%    25%    50%    75%    90%    95%    99%    max
------------------------  ----------  -----  ----  ----  -----  -----  -----  -----  -----  -----  -----  -----
src_any                   172.354        56    71    83     88    104    126    186    305    424    740   2322
src_self                    3.06404       1     1     1      1      1      1      3      6     10     24    137
src_click_to_click         20.474         0     0     2      9     10     10     21     41     63    133    553
src_click_to_cart_or_buy   20.1775        0     0     0      3     10     10     20     42     66    139    569
src_cart_to_cart            4.66161       0     0     0      0      0      0      0     20     29     80    657
src_cart_to_buy             3.31507       0     0     0      0      0      0      0      7     22     63    517
src_buy_to_buy              0.324088      0     0     0      0      0      0      0      0      0      9    266
src_w2vec_all              42.7851        0     0     0     20     20     20     48     91    132    243    858
src_w2vec_1_2              40.9391        0     0     0      0     20     20     41     92    133    242    861
src_pop_cl50               56.9076       32    34    35     37     44     54     65     86     86     86     86
```
