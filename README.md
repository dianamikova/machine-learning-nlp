*A collection of machine learning projects with a focus on Natural Language Processing (NLP), implemented in Python.*


# Sentiment Analysis Experiments (`sentiment-analysis/`)

*An exploratory NLP notebook using HuggingFace Transformers to study how sentiment models
and language models behave across different contexts and geographies.*

Results analysis: Qualitative discussion of model failures and biases observed across all experiments.

Libraries

`transformers` `torch` `plotly` `pandas`


# Sentiment Classification (`sentiment-classification/`)

*A text classification project that predicts the sentiment (positive/negative) of input text using the **IMDb reviews** dataset.*

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

Libraries

`matplotlib` `numpy` `scikit-learn` `scipy` `torch` `transformers`

# Evaluating and Extending an RNN based Part-of-Speech Tagger

An RNN-based part-of-speech tagger evaluated and extended across 11 Universal Dependencies treebanks in 7 languages (English, Swedish, Danish, Finnish, Czech, Romanian). 
The base RNN is refactored into a scikit-learn–style RNNPosTagger class, then extended with GRU cells, bidirectionality, dropout, and token-masking augmentation. 
Best config: **BiLSTM, dropout = 0.3.**

**Baseline** = *per-token majority*: predict each word's most frequent training tag (lowercased), falling back to the global most-frequent tag for unseen words. 
Purely lexical, no context — already strong because most words are unambiguous.

Dataset: Universal Dependencies (UD) treebanks — annotated corpora where each word is tagged with its part of speech.


| # | Approach | Accuracy |
|---|----------|----------|
| 1 | Global majority baseline | 16.4% |
| 2 | GRU (unidirectional) | 84.1% |
| 3 | LSTM (unidirectional) | 84.8% |
| 4 | Per-token majority baseline | 86.4% |
| 5 | BiLSTM + dropout 0.3 | 88.5% |
| 6 | BiLSTM (no dropout) | 88.7% ✅ |

## Cross-language (Baseline vs. BiLSTM + dropout 0.3)

| # | Treebank (lang / genre) | Baseline | BiLSTM |
|---|-------------------------|----------|--------|
| 1 | en_ewt (Eng / web) | 86.4% | 91.8% |
| 2 | en_gum (Eng / mixed) | 86.3% | 92.2% |
| 3 | en_lines (Eng / literary) | 88.2% | 93.2% |
| 4 | sv_talbanken (Swe / news) | 87.5% | 92.1% |
| 5 | sv_lines (Swe / literary) | 87.1% | 91.3% |
| 6 | da_ddt (Dan / news) | 85.9% | 89.9% |
| 7 | fi_tdt (Fin / news) | 87.6% | 87.4% |
| 8 | fi_ftb (Fin / grammar) | 82.5% | 83.6% |
| 9 | cs_cac (Cze / news) | 92.4% | 94.2% |
| 10 | cs_fictree (Cze / fiction) | 86.8% | 93.4% |
| 11 | ro_rrt (Rom / news) | 90.7% | 93.8% |

BiLSTM beats the baseline on 10 of 11 (Finnish fi_tdt the lone exception).

## Key ablations

| Experiment | Finding |
|-----------|---------|
| LSTM vs. GRU | Roughly tied (e.g. Eng 84.8 / 84.1) |
| Bidirectionality | Big gains (Eng 84.9 → 88.7) except Finnish |
| Dropout | Best at 0.3 (88.5% test) |
| Augmentation | *Hurts* English, *helps* Finnish (high OOV) |
| Model complexity | 64.7% → 84.0% then diminishing returns |
| UPOS vs XPOS | Finer tagsets much harder (Czech 92.9% → 82.1%, 1176 tags) |

## Libraries

`matplotlib` `numpy` `scikit-learn` `torch` `tqdm`

