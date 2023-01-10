# OTTO – Multi-Objective Recommender System

## Description of the competition
* Kaggle: OTTO – Multi-Objective Recommender System]: https://www.kaggle.com/competitions/otto-recommender-system  
* Data documentation from provider: https://github.com/otto-de/recsys-dataset

## What's in the data?
u* **12.9M** real-world anonymized user sessions ()
* **220M** events, consisting of clicks, carts and orders
* **1.8M** unique articles in the catalogue

## Solution
High level:  
Corpus of 1.8M articles > Retrieve top ~1K > Rank > Submit top 20

### Retrieval
Train word2vec model(s), where the sequence of actions is a sentence and the 
aid values are words. Each aid gets an embedding vector. 
For each aid from truncated session (start from last), retrieve top N similar 
aids (by cosine/euclidian similarity) using KNN. Add these top N aids to the 
pool of candidates to be later ranked.
We can also train several models:
- using all actions
- using clicks
- using carts
- using orders

Train doc2vec model(s), where a session is a sentence. 
Each session gets an embedding vector. Find top N similar sessions, collect the 
remainders of these sessions, retrieve most frequent aids.

Train a seq2seq model (LSTM). Predict next sequence. Use only cards+orders.


### Features Engineering
#### Statistics by aid
- click popularity (% out of total sessions) during last 24h, week, all-time
- carts popularity (% out of total sessions) during last 24h, week, all-time
- order popularity (% out of total sessions) during last 24h, week, all-time
- prob of click if clicked before
- prob of carts if clicked before
- prob of carts if added to cart before
- prob of order if added to cart, but not ordered yet
- prob of order if added to cart and ordered, then added to card again
- prob of order if ordered before

#### Statistics by user session
- duration
- number of actions during last 24h, week, all-time
- number and % of clicks during last 24h, week, all-time
- number and % of carts during last 24h, week, all-time
- number and % of orders during last 24h, week, all-time
- % of days active


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
