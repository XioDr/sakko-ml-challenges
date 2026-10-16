# Sakko ML Challenges

**1st place** at the 9th ICDS Data Science & AI Hackathon (Sakko): *Prediction and intelligent prioritization of manufacturing anomalies*. Code, documentation and reference outputs for the three challenges, all built on one shared feature pipeline over quality-notification data (NCR rework duration).

[Competition description](https://sakkoo.ai/competitions/6/description/) · Author: [Danial Rahmani](https://github.com/XioDr), Founder & CEO of [Alacrity](https://www.alacrity.agency/)

## Results

Scores are the public-leaderboard values recorded by the competition platform; each is tied to a named, hash-checked output file. Hidden-test labels were not available to us, so none of these numbers was recomputed independently.

| Challenge | Task | Metric | Public score | Reference output |
| :-- | :-- | :-- | --: | :-- |
| 1 | Binary: `short` (up to 480 min) vs `long` | Balanced accuracy | 0.837 (target-only path) | `challenge_1/submission_package_c1/reference/submission_model_v42_0.837.csv` |
| 1 | Competition result | Balanced accuracy | 0.844 | `challenge_1/hour_v1/submission_hour_v80_mcmc_c2mix_k3730.csv` |
| 2 | Four duration classes | Balanced accuracy | 0.674 | `challenge_2/submission_challenge2_v120_prior_P.csv` |
| 3 | Regression of rework minutes | MAE (lower is better) | 42.094 | `challenge_3/v93_lean/submission_challenge3_v114_yb300_nov08.csv` |

## Approach

- **Shared features.** A 96-feature pipeline (`shared_features/build_features.py`) built from the item, first-activity, production-tree and material tables, plus text embeddings of item and activity descriptions.
- **Challenge 1.** Models trained on the binary target; the competition-result file additionally mixes in a signal from challenge 2.
- **Challenge 2.** Base models trained on the four-class target. The final decision is a frozen posterior fitted with public-leaderboard feedback, which can be reconstructed exactly.
- **Challenge 3.** A distributional duration model that uses signals from challenges 1 and 2, a temporal correction, and a decision rule matched to the MAE metric.

## Transparency note

The C1 competition result, C2 v120 and C3 v114 use cross-challenge signals and/or public-leaderboard feedback for model selection. This is disclosed in full in the technical report so that reviewers can judge compliance. The full report (Persian, right-to-left) is in [`deliverables/organizer/TECHNICAL_REPORT_FA.pdf`](deliverables/organizer/TECHNICAL_REPORT_FA.pdf), with a Markdown version next to it. Parts of the research, coding and documentation were done with LLM-based assistants; reported scores come from the platform's evaluation of the CSV files, not from an LLM.

## Repository layout

- `challenge_1/`, `challenge_2/`, `challenge_3/`: training, validation and prediction code per challenge, with reports (mostly Persian, `*_FA.md`).
- `shared_features/`: the shared feature pipeline.
- `research_pipeline/`, `research/`: experiment scripts, sweeps and research logs.
- `deliverables/`: technical report and packaging scripts.
- `DATA_OVERVIEW.md`: description of the competition data tables (Persian).

## Not included

The competition data (raw CSVs, 4096-dimensional text embeddings), feature caches and trained models belong to the organizer or are very large, so they are not in this repository. Place the official data under each challenge's `data/` folder before running anything.

## Setup

```bash
python -m venv .venv
pip install -r challenge_1/requirements.txt
pip install -r challenge_2/requirements_multiclass.txt
```

Tested with Python 3.12. Some challenge 2 and 3 training steps expect an NVIDIA GPU.
