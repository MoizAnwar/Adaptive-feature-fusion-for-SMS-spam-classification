# Adaptive Feature Fusion for SMS Spam Classification

Harsh Jayeshbhai Goyani · Mohammed Abdul Moiz Anwar

## Project Overview

This project asks whether **input-dependent (adaptive) feature fusion** improves SMS spam
classification over single representations and simple concatenation.

It compares five systems on a fixed split of the SMS Spam Collection:

1. TF-IDF + Logistic Regression
2. Mean GloVe (100-d) + Logistic Regression
3. Frozen DistilBERT + Logistic Regression
4. Simple concatenation (TF-IDF + GloVe + DistilBERT) + Logistic Regression
5. **Proposed adaptive fusion classifier** (learned gate)

**Headline result (single saved run, seed 42):**

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Frozen DistilBERT + LR | 0.9928 | 0.9847 |
| Simple concatenation + LR | 0.9916 | 0.9820 |
| Adaptive fusion (proposed) | 0.9916 | 0.9818 |
| TF-IDF + LR | 0.9857 | 0.9691 |
| GloVe mean + LR | 0.9498 | 0.9044 |

Adaptive fusion beats TF-IDF and GloVe but does **not** beat frozen DistilBERT or
concatenation. This is one run, so it does not show a statistically significant or
general advantage for any model.

## Dataset and Split

- Source: `ucirvine/sms_spam` (5,574 messages; label 0 = ham, 1 = spam)
- Stratified split, seed 42: 3,901 train / 836 dev / 837 test
- Test set: 725 ham, 112 spam
- Text cleaning: lowercase and whitespace normalisation

## Model Design

- **TF-IDF:** 5,000 unigram and bigram features, min_df=2, sublinear TF
- **GloVe:** `glove-wiki-gigaword-100`, mean of known-token vectors, standardised
- **DistilBERT:** `distilbert-base-uncased`, frozen, masked mean pooling, 768-d, max 128 tokens
- **Baselines:** balanced Logistic Regression (max_iter=2000)
- **Adaptive model:**
  - Each representation is projected to 256-d.
  - A gate network reads TF-IDF and GloVe features and outputs a weight.
  - The weight mixes projected DistilBERT against projected GloVe.
  - The TF-IDF projection is added separately.
  - A LayerNorm + ReLU + Dropout classifier head produces the prediction.
  - Training uses weighted cross-entropy and AdamW (lr 1e-3).
  - The checkpoint is selected by dev macro-F1, with patience 5.
- The transformer is never fine-tuned or skipped.

## Execution Flow

```
1. Load dataset (ucirvine/sms_spam)
        |
2. Clean text -> stratified train / dev / test split (seed 42)
        |
3. Extract features
   |-- TF-IDF (fit on train only)
   |-- Mean GloVe (scaler fit on train only)
   |-- Frozen DistilBERT embeddings
        |
4. Train baselines (Logistic Regression)
   TF-IDF | GloVe | DistilBERT | Concatenation
        |
5. Train adaptive fusion model
   projections -> gate -> fused vector -> classifier
   select best epoch on dev macro-F1 -> restore checkpoint
        |
6. Evaluate all models on the test set
   accuracy, macro-F1/precision/recall, training time
        |
7. Test analysis
   confusion matrix, error analysis, gate analysis
        |
8. Save outputs
   main_results.csv, adaptive_hybrid_test_analysis.csv
```
## Files

| File or folder | Purpose |
|---|---|
| `code/adaptive_hybrid_sms_spam_reviewed.ipynb` | Complete experiment code and preserved historical outputs, with explanatory annotations. |
| `data/main_results.csv` | Supplied five-model results. |
| `data/adaptive_hybrid_test_analysis.csv` | Supplied 837 test messages, labels, predictions and gate analysis. These messages originate from the attributed SMS benchmark. |
## Check the saved results

## Key Findings

- The adaptive model classified 830 of 837 test messages correctly (107/112 spam, 723/725 ham).
- Spam precision is 98.17% and spam recall is 95.54%.
- Frozen DistilBERT leads the adaptive model by about 0.29 macro-F1 points.
- Adaptive fusion is much slower to train (about 29.8 s, against 0.25 s for DistilBERT + LR), and this excludes feature extraction.
- Gate values are descriptive. They are not causal explanations or a percentage contribution.

## Limitations

- One seed, no significance tests, no gate ablations.
- The hybrid uses a nonlinear head and the baselines use linear classifiers, so the comparison is between complete systems, not the gate alone.
- Cross-split duplicate overlap is unverified (10 repeated test rows).
- Exact package versions and hardware were not recorded, and full training has not been re-validated.
- Next steps: group duplicates before splitting, repeat over several seeds, and test gate ablations.

