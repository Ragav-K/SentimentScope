# SentimentScope: Sentiment Analysis with a Transformer

A transformer model trained **from scratch** (PyTorch) to classify IMDB movie reviews as positive or negative. Built as the SentimentScope project for CineScope's recommendation system.

## Results

| Metric | Value |
|---|---|
| Test accuracy (25,000 reviews) | **78.04%** |
| Best validation accuracy | 80.04% (epoch 4) |
| Parameters | ~4.7M |

Training ran for 5 epochs; the weights from the best validation epoch were kept.

## Model

- Subword tokenization with the `bert-base-uncased` tokenizer (tokenizer only, no pretrained weights), max length 128
- 4 transformer blocks, 4 attention heads, 128-dim embeddings
- Mean pooling over the token embeddings, followed by a linear classification head (2 classes)
- AdamW, learning rate 3e-4, batch size 32

## Contents

| File | Description |
|---|---|
| `SentimentScope.ipynb` | Full notebook: data loading and exploration, `IMDBDataset`, model, training, testing, inference, conclusion |
| `sentimentscope_checkpoint.pt` | Trained checkpoint (weights + config) |
| `requirements.txt` | Python dependencies |

## Usage

1. Install dependencies: `pip install -r requirements.txt` (use a CUDA build of PyTorch to train on a GPU).
2. Download the [IMDB dataset](https://ai.stanford.edu/~amaas/data/sentiment/) (`aclImdb_v1.tar.gz`) into the project folder.
3. Run `SentimentScope.ipynb` top to bottom. The first code cell extracts the archive.

### Inference with the saved checkpoint

The notebook defines `load_model()` and `predict_sentiment()`:

```python
model, tokenizer, max_length = load_model("sentimentscope_checkpoint.pt", device)
predict_sentiment(["A beautiful, moving film.", "Boring and far too long."],
                  model, tokenizer, device, max_length)
```

This returns the predicted sentiment and the probability of each class for every review.

## Notes

Reviews are truncated to 128 tokens, so the model only sees the start of most reviews, which limits accuracy. Longer context, a larger model, and a padding mask are the most promising improvements.
