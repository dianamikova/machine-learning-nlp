Model Architecture

The SentimentClassifier class loads a pre-trained encoder using AutoModel.from pretrained()
following the HuggingFace AutoModels API. A custom classifier head (nn.Linear) is
added on top of the frozen encoder, as the assignment requires the classifier to be designed
independently rather than using a ready-made pipeline such as AutoModelForSequence-
Classification.
The encoder was frozen to avoid out-of-memory errors; BERT has 110 million param-
eters, and storing gradients for all of them would exhaust available GPU memory. As a
consequence, with only 1538 trainable parameters in the classifier head, the model can-
not overfit, which is why validation accuracy consistently matched or exceeded training
accuracy throughout training.

Sequence Length

Initially avg len=253 was used (average review length + 20 token buffer). After ob-
serving slow training and high initial loss, the value was reduced to 186, cutting ap-
proximately 26 % of tokens. This reduces attention computation by approximately 54
%, which matched a 35 % speed improvement in practice. The assumption behind this
decision is that the beginning of a review usually contains enough sentiment signal to
determine polarity - readers typically establish their opinion early.

Results

The monolingual BERT outperforms its multilingual counterpart by 13.4%, which is ex-
pected since the multilingual model distributes its capacity across 104 languages, resulting
in weaker English-specific representations. Notably, distilroberta-base outperforms
its full counterpart roberta-base (79.1% vs 76.1%), consistent with the DistilBERT
paper’s claim [4] that distilled models retain 97% of language understanding while be-
ing 60% faster. By the third epoch, distilroberta-base also overtakes the previously
strongest model, BERT — it would be interesting to see how this trend develops with
more epochs.
Logistic Regression outperforms all transformer models, which can be explained by
two factors: full access to review text without truncation, and the frozen encoder limiting
the transformer’s ability to adapt to the task. Full fine-tuning of the encoder would likely
reverse this result.
