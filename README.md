# Arabic Hotel Reviews — Sentiment Analysis (Frozen AraBERT + Custom Attention)

3-class sentiment classification (Negative / Neutral / Positive) on Arabic hotel reviews, using a frozen AraBERT backbone with a custom attention layer and classifier head built from scratch.

## Overview

- **Task:** Text classification (3-class sentiment analysis)
- **Domain:** Arabic hotel reviews (hospitality)
- **Dataset:** HARD — Hotel Arabic-Reviews Dataset (93,700 Booking.com reviews), refined to a QA-audited subset of 22,500 samples
- **Backbone:** `aubmindlab/bert-base-arabertv02` (AraBERT v2), **frozen**
- **Custom components:** Bahdanau-style additive attention layer + linear classifier head (768→256→3), trained from scratch on top of the frozen backbone
- **Bonus:** Synthetic data augmentation (1,500 LLM-generated hard examples: sarcasm, mixed sentiment, dialect-heavy negative, subtle positive) to strengthen weak spots

## Pipeline

| Phase | Description |
|-------|-------------|
| 1. Data engineering | Load HARD dataset, EDA (class balance, review length), 11-step Arabic text cleaning/normalization pipeline (diacritics, alef/ya/ta-marbuta normalization, HTML/URL/emoji stripping, AraBERT preprocessor) |
| 1b. Label QA | 22,500 samples manually audited; 5,664 relabeled (mislabeled or ambiguous), 16,658 kept as-is |
| 2. Architecture | AraBERT tokenizer → frozen AraBERT embeddings → custom additive attention → linear classifier |
| 3. Training | Class-weighted cross-entropy loss, AdamW (only attention + classifier params), ReduceLROnPlateau scheduler, early stopping on validation loss |
| 3b. Evaluation | Accuracy, macro F1 (key metric for imbalance), per-class precision/recall/F1, confusion matrix, misclassification review, attention-weight visualization |
| Bonus | Inject 1,500 synthetic hard examples, recompute class weights, retrain and compare |

## Model architecture

```
Text → AraBERT Tokenizer → Frozen AraBERT (bert-base-arabertv02)
                                    │
                          token embeddings (batch, seq_len, 768)
                                    │
                   Custom Additive Attention (768 → 256 → 1, softmax over tokens)
                                    │
                          context vector (batch, 768)
                                    │
                   Linear(768→256) → ReLU → Dropout(0.3) → Linear(256→3)
                                    │
                                 logits
```

Only the attention layer and classifier head are trainable; the AraBERT backbone stays frozen throughout.

## Key configuration

| Parameter | Value |
|-----------|-------|
| Model | `aubmindlab/bert-base-arabertv02` |
| Max sequence length | 128 |
| Attention dim | 256 |
| Dropout | 0.3 |
| Batch size | 32 |
| Learning rate | 2e-4 (custom layers only) |
| Max epochs | 20 (early stopping, patience 5) |
| Train / val / test split | 85% / 15% of train / 15% held-out |
| Loss | Class-weighted cross-entropy (`balanced` weights) |

## Setup

```bash
pip install datasets transformers arabert pyarabic scikit-learn matplotlib seaborn tqdm torch
```

Requires a GPU-enabled environment (Colab/Kaggle recommended) for practical training time.

## Data files expected

- `filtered_qa_results.csv` — QA-corrected 22,500-sample training set (columns: `text`, `label`, `sentiment`)
- `cleaned_arabic_hotel_reviews_1500.json` — synthetic augmentation examples (bonus phase)

Update the paths in the notebook (`CSV_PATH`, the synthetic JSON path) to match wherever these files are uploaded in your environment.

## Usage

1. Run the installation and imports cells.
2. Load and clean the dataset (Phase 1).
3. Build the model and training components (Phase 2).
4. Run the training loop and evaluate on the held-out test set (Phase 3).
5. Optionally run the bonus phase to train with synthetic data augmentation and compare metrics.
6. The final cell saves `model_config.json`, the AraBERT tokenizer, and the trained weights (`best_model.pt`) for reproducible inference via a `predict_sentiment()` helper.

## Notes

- Macro F1 is the primary evaluation metric since the dataset is imbalanced (Positive reviews dominate).
- Attention weights are visualized per-review to inspect which words drove each prediction.
- The 11-step Arabic cleaning pipeline is applied identically to both the original and synthetic data to keep preprocessing consistent.
