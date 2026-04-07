A collection of machine learning projects with a focus on Natural Language Processing (NLP), implemented in Python.


Sentiment Classification (sentiment-classification/)
A text classification project that predicts the sentiment (e.g. positive/negative) of input text.

Dataset: IMDb reviews
Approaches:
  1. Random Baseline
  2. TF-IDF + SGD
  3. TF-IDF + Logistic Regression
  4. bert-base-uncased
  5. bert-base-multilingual-cased
  6. distilroberta-base
  7. roberta-base
Libraries: matplotlib.pyplot, time, os.path, tarfile, urllib.request, os, numpy, sklearn.feature_extraction.text, sklearn.base, scipy.sparse, sklearn.linear_model, random, torch.utils.data, transformers, gc, torch, torch.nn
Accuracies:
  1. Random baseline: 49.6%
  2. Test accuracy: 61.2%
  3. Test accuracy: 88.3%
  4. Highest accuracy in 3/3 epoch: 75.2%
  5. Highest accuracy in 3/3 epoch: 61.4%
  6. Highest accuracy in 3/3 epoch: 76%
  7. Highest accuracy in 3/3 epoch: 69.1%
