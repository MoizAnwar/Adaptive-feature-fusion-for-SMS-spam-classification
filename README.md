# Adaptive feature fusion for SMS spam classification

Harsh Jayeshbhai Goyani and Mohammed Abdul Moiz Anwar

This empirical project compares TF-IDF, mean GloVe, frozen DistilBERT, simple concatenation and an adaptive fusion classifier on a fixed SMS Spam Collection split. The supplied run uses 3,901 training, 836 development and 837 test examples. Frozen DistilBERT plus logistic regression has the highest recorded macro-F1 (0.984654); adaptive fusion reaches 0.981754. This is one saved run, not evidence of a statistically significant or general performance advantage.

## Files

| File or folder | Purpose |
|---|---|
| `code/adaptive_hybrid_sms_spam_reviewed.ipynb` | Complete experiment code and preserved historical outputs, with explanatory annotations. |
| `assets/data/main_results.csv` | Supplied five-model results. |
| `assets/data/adaptive_hybrid_test_analysis.csv` | Supplied 837 test messages, labels, predictions and gate analysis. These messages originate from the attributed SMS benchmark. |
| `requirements.txt` | Experiment dependency names; not a tested or historical lockfile. |
| `code/audit_saved_results.py` | Standard-library check of supplied predictions and metrics; no model training or downloads. |
| `audit/recomputed_metrics.json` | Output from the saved-result check. |
| `audit/EXPERIMENT_EVIDENCE.md` and `audit/CLAIM_TRACEABILITY.md` | Method details, evidence and limits. |
| `REFERENCES.md` | Fourteen research and resource references. |
| `code/CODE_PROVENANCE.md` and `code/THIRD_PARTY_NOTICES.md` | Source attribution, AI-assistance record and resource notices. |

## Check the saved results

From the repository root:

```sh
python3 code/audit_saved_results.py
```

No third-party Python packages are needed for this check. Expected hybrid confusion matrix, with true labels in rows and predictions in columns, is `[[723, 2], [5, 107]]`. Accuracy is 0.9916367981; macro-F1 is 0.9817540866. The script also verifies the saved heuristic columns and identifies 10 repeated test rows beyond first occurrences. Baseline predictions were not supplied, so their reported scores cannot be independently reconstructed here.

## Run a new experiment

The original notebook reports Python 3.10. Its exact package versions, resource revisions, hardware and split IDs were not recorded. Full training has not been rerun or validated for this repository package. Use a separate environment and keep the supplied CSVs unchanged:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyter notebook code/adaptive_hybrid_sms_spam_reviewed.ipynb
```

Run the notebook cells in order. The notebook retains two historical unpinned installation commands, including a package upgrade; they may change the environment. Resolve and record a working package set before claiming exact reproduction. Save `python --version`, `python -m pip freeze`, dataset/model revision identifiers, split IDs and a checkpoint for the new run. Keep new results separately and compare them with the supplied record. By default, the notebook writes CSV files into its kernel working directory.

The first full run downloads `ucirvine/sms_spam`, `glove-wiki-gigaword-100` and `distilbert-base-uncased`. These resources and pretrained weights are not bundled. The notebook uses seed 42 and frozen DistilBERT features. The gate combines projected DistilBERT and GloVe and adds a separate TF-IDF projection. It does not skip the transformer or fine-tune it.

## Interpretation and provenance

All supplied notebook execution results and both CSV files are retained. Code labels were previously corrected from BERT to DistilBERT without changing the model logic. New Markdown, evidence checks, poster wording and repository preparation used AI assistance. Original notebook authoring history and any earlier AI or borrowed-code use have not been independently established. The students should review the work and follow their examiner's disclosure requirements. See `code/CODE_PROVENANCE.md` for the detailed record.

Training time fields exclude feature extraction and are not end-to-end runtime measurements. Cross-split duplicate overlap remains unverified. There are no additional seeds, gate ablations or significance tests in the supplied evidence. No new experiment has been substituted for the historical results.

## Resources and licensing

Dataset, model, library and research sources are listed in `REFERENCES.md` and `code/THIRD_PARTY_NOTICES.md`. Resource terms are distinct from a project-wide code licence; no new blanket licence has been assigned. Pretrained weights, dependency source trees, credentials, signed declarations, poster drafts and machine-specific build files are not part of this repository package.
