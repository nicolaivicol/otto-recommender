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

### Ranking
Train 3 separate LightGBM rankers for clicks, carts and orders.  
Trained only on first 500K test session (due to memory limits).

### Results
Recall@20 on local test sessions: **0.5620** 
  * clicks: 0.4925 
  * orders: 0.6508  
  * carts: 0.4074

Recall@20 on LB: **0.565** (without re-counting co-events)
