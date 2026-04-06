# Learning to Rank with Gradient Boosting: MSR LETOR, WikiIR, and MIRAGE

**Python 3.11+ | MIT License**

[![CatBoost](https://img.shields.io/badge/CatBoost-1.2.10-blue)](https://catboost.ai/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6.0-green)](https://lightgbm.readthedocs.io/)

This repository contains a complete implementation of learning-to-rank (LTR) experiments for the HA3 assignment, including:

- **Pointwise & pairwise ranking** with CatBoost (YetiRank) and LightGBM (LambdaMART).
- **MSR LETOR (MQ2007)** benchmark – applying CatBoost to a standard LETOR collection.
- **Internet Math 2009** – reformatting to LETOR format, comparison of two rankers (NDCG@10).
- **WikiIR (en1k)** – improving BM25 with extra features (query length, term distance, TF mean) and training a reranker.
- **BM25 reconstruction** – learning to predict BM25 scores from its core components (TF, IDF, document length).
- **MIRAGE** – passage ranking with external pageview & backlink features (Wikimedia API) and stratified train/test split by question source.
- **Feature importance analysis** and **NDCG@k** evaluation via `ir_measures`.

All code is self-contained in a single Jupyter Notebook and uses only open‑source libraries.

---

## 📊 Key Results

| Collection          | Model              | Features                     | NDCG@10 |
|---------------------|--------------------|------------------------------|---------|
| Internet Math 2009  | CatBoost (YetiRank)| 145 LETOR features           | 0.8207  |
| Internet Math 2009  | LightGBM (LambdaMART)| 145 LETOR features         | 0.8178  |
| WikiIR (train)      | BM25 (baseline)    | –                            | 0.0000* |
| WikiIR (train)      | LTR (CatBoost)     | query_len, doc_len, common_words, min_dist, tf_mean | 0.0000* |
| WikiIR (reconstruction) | CatBoostRegressor | tf_mean, bm25_score, doc_len | Pearson=0.999 |
| MIRAGE              | CatBoost (YetiRank)| text stats + pageviews/backlinks (simulated) | 0.8706 |

> *Zero NDCG on WikiIR occurs because relevant documents for test queries are not present in the top‑100 BM25 results. This is a known issue with the collection split and requires a different retrieval depth or forced inclusion of relevant docs.

---

## 📂 Data Sources

| Collection | Documents | Queries | Relevance | Format |
|------------|-----------|---------|-----------|--------|
| MSR LETOR (MQ2007) | ~700k | 1,700 (Fold1) | graded (0-2) | LETOR (libsvm) |
| Internet Math 2009 | ~77k | 70 (train) | graded (0-4) | LETOR |
| WikiIR (en1k) | 369,722 | 1,444 training / 100 test | binary | CSV + TREC qrels |
| MIRAGE | 37,800 passages | 7,560 | binary (oracle) | JSON |

---

## 🧠 Method Overview

```mermaid
flowchart TD
    A[Query & Document] --> B[Feature Extraction]
    B --> C{BM25 baseline?}
    C -->|Yes| D[Compute BM25 score]
    C -->|No| E[Extract additional features:<br/>query_len, doc_len,<br/>common_words, min_dist,<br/>pageviews, backlinks]
    D & E --> F[Create query-doc vector]
    F --> G[Train CatBoostRanker / LGBMRanker]
    G --> H[Predict on test queries<br/>(top-100 BM25 candidates)]
    H --> I[Evaluate NDCG@10]
