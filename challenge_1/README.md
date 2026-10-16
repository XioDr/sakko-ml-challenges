# Sakko binary NCR challenge

Predict `short` (recorded rework <= 480 minutes) or `long` (> 480 minutes).
The only target used is `data/target_binary_train.csv`.

## Run

From this directory, with Python 3.12 and the packages in `requirements.txt`:

```powershell
python -u run_challenge.py
python predict.py
```

The first command rebuilds features, validates the approach, fits final models,
and writes `output/submission.csv`. It overwrites the generated output artifacts.
The second loads saved models and reproduces the submission; if an existing
submission differs, it raises an error before overwriting it.

## Data and features

- Training: 73,239 items; test: 8,323 items, kept in original CSV order.
- Join targets and initial activities using `(Notification, Item)`.
- Optional tables are left joined; missing optional records do not remove items.
- Join material metadata separately for `Material` and `Assembly`.
- Aggregate MAS trees by `tree_id` before joining; also aggregate matching material
  node counts and depths within each tree.
- Derive date, clock, categorical group/code pairs, and missingness features.
- Keep zeros distinct from missing values. Remove the entirely blank WBS column.
- For each text family, preserve the mapping of Parquet row i to NPY row i.
- Fit PCA on up to 18,000 unique available embedding rows associated with the
  training portion only: 48 item-text, 32 activity-text, and 24 material-text components.
- Fit class-balanced logistic regression on normalized 4,096-dimensional item text.
  Missing item text falls back to the structured model probability in the blend.
- No target values from other challenges, test labels, or external training data are used.

## Temporal evaluation

1. Train before 2023 and validate on 2023. Remove historical rows whose notification
   also appears in the validation year. Fit dimensionality reduction on training rows.
2. On 2023, select CatBoost iteration counts, blend weights, and prediction threshold.
   This is a tuning score, not an untouched test estimate.
3. Train before 2024 and evaluate on 2024, again purging shared notifications.
   Keep the 2023 iteration counts, blend, threshold, and other hyperparameters fixed.
4. Refit PCA and models on all labeled training data and predict the provided test set.

Supplementary tables are the supplied snapshots; an earlier historical snapshot
cannot be reconstructed. Validation results therefore reflect this dataset's
permitted information and cannot establish real-time production performance.

## Outputs

- `output/submission.csv`: upload this one-column file to the binary challenge.
- `output/test_predictions_with_ids.csv`: predictions, model scores, and original item keys for auditing.
- `output/validation_metrics.json`: class recall, balanced accuracy, AUC, confusion matrices.
- `output/2023/validation_predictions.csv` and `output/2024/validation_predictions.csv`.
- `output/selection.json`: settings chosen using 2023.
- `output/run_metadata.json`: selected settings, feature names, versions, final class counts.
- `output/full/`: trained CatBoost, logistic regression, and PCA models.

The scores are class-weighted model outputs, not calibrated estimates of class prevalence.
The official test score requires submission to the competition site.

## Input verification

The ZIP passed a full CRC check. Its SHA-256 is:
`fa903262ef6b9d63550984002a058487accd92645ae3c965374453bad01909de`.
The training pipeline checks unique item/target keys, complete label matching,
no train-test item-key overlap, eligible activity delays, embedding dimensions,
finite embedding values, activity embedding joins, and final submission shape/order.
