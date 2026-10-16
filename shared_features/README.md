# Shared feature pipeline (96 features)

`build_features.py` builds one feature table per split from the raw common
data (`data/`). No target values are used; every train-fit statistic
(frequency encodings, rare-category thresholds, PCA) is fit on
`items_train.csv` (or the matching train-side table) only, then applied to
both splits. Safe to join to any of the three targets or to reuse across CV
folds.

Run:
```
python build_features.py
```
Outputs: `features_train.parquet`, `features_test.parquet` (keyed by
`Notification`, `Item`), `feature_list.txt`.

## Categories (96 total)

| # | group | count | notes |
|---|---|---|---|
| 1 | temporal | 10 | year/month/quarter/dow/hour, weekend flag, cyclical sin/cos |
| 2 | defect group/code | 16 | freq + rare flag for problem/obj.part/cause group+code, group→code cardinality, any-missing flag |
| 3 | severity/priority/prevention | 10 | freq + rare, priority one-hot (4 cats) + missing flag |
| 4 | numeric quantities | 8 | log1p defect counts, internal/external ratio, complaint/reference (shared missing flag) |
| 5 | material joins | 16 | `Material`→material_general and `Assembly`→material_general, 8 features each (weight/volume/type freq/group freq/qm flag/base-material-missing/found) |
| 6 | mas_trees aggregation | 8 | per-`tree_id`, computed from the matching split's own tree file before joining |
| 7 | first activity | 6 | delay_h (+log1p), has-activity flag, activity code/group freq, activity sequence number |
| 8 | notification structure | 6 | items per notification, this item's position/fraction, single-item flag, date span, distinct-severity count |
| 9 | text embeddings → PCA | 16 | item_longtext (8), first-activity text (4), material_desc (4); PCA fit on up to 30k train rows |

## Known data behavior (not bugs)

- `material_found` / `assembly_found` are ~74% / ~22% respectively — most
  distinct `Material` ids do exist in `material_general`, but the
  higher-frequency ids in `items` skew toward the ones that don't. The data
  dictionary explicitly warns a matching record isn't guaranteed.
- `has_first_activity` ≈ 42%, matching the dictionary's stated first-activity
  coverage.
- Sibling-item **target** history (e.g. other items under the same
  Notification) is intentionally NOT included here — that needs leakage-safe
  out-of-fold treatment per challenge/target and should be added per-challenge
  on top of this table, not baked into the shared pipeline.
