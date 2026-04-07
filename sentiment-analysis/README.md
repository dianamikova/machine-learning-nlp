## Sentiment Analysis Experiments (`sentiment-analysis/`)

An exploratory NLP notebook using HuggingFace Transformers to study how sentiment models
and language models behave across different contexts and geographies.

### Experiments

**1. In-domain experiment**
Runs `distilbert-base-uncased-finetuned-sst-2-english` on custom movie review sentences
to test how well the model handles nuanced or idiomatic language (e.g. *"not my cup of tea"*).

**2. Out-of-domain experiment**
Applies the sentiment pipeline to country-templated sentences (e.g. *"This movie was filmed in {country}."*)
and visualizes results as an interactive world choropleth map using Plotly.

**3. Custom queries**
Extends the geographic analysis to user-defined sentence templates
(e.g. *"I am going to work in {country}."*) to probe implicit country bias in the model.

**4. GPT-2 perplexity analysis**
Uses GPT-2 to measure how *surprising* country-templated sentences are,
visualizing perplexity scores on a world map to reveal which country names feel
more or less natural to the model.

**5. Results analysis**
Qualitative discussion of model failures and biases observed across all experiments.

Libraries

`transformers` `torch` `plotly` `pandas`
