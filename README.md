*A collection of machine learning projects with a focus on Natural Language Processing (NLP), implemented in Python.*

## Projects

### 1. Sentiment Classification (`sentiment-classification/`)

A text classification project that predicts the sentiment (positive/negative) of input text using the **IMDb reviews** dataset.

Results

| # | Approach | Accuracy |
|---|----------|----------|
| 1 | Random Baseline | 49.6% |
| 2 | TF-IDF + SGD | 61.2% |
| 3 | TF-IDF + Logistic Regression | 88.3% ✅ |
| 4 | bert-base-uncased | 75.2% |
| 5 | bert-base-multilingual-cased | 61.4% |
| 6 | distilroberta-base | 76.0% |
| 7 | roberta-base | 69.1% |

Libraries & Tools

`matplotlib` `numpy` `scikit-learn` `scipy` `torch` `transformers`
