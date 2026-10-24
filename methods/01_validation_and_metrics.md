# 01. Validation and metrics

## What
- Folds: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` over the 6,431 train pets, stratified by AdoptionSpeed.
  The same folds are used by every model and by the neighbour features, so out-of-fold (OOF) predictions of all models
  can be stacked without leakage.
- Regression-style outputs (a number around 1..4) are turned into classes by three cut points.
- Three ways to score OOF predictions:
  1. tuned: cut points found by coordinate search maximising QWK on all OOF predictions (`fit_cuts` in `_work/feats.py`
     and in the notebook). Slightly optimistic.
  2. nested: cut points fitted on 4 folds, applied to the 5th fold, then QWK over all rows.
  3. dist ("train-share cuts"): inside each validation fold the cut points are the quantiles of that fold's predictions
     at the cumulative train class shares (18.6%, 46.2%, 66.8%). This is the same rule used for the test set.

## Why
- QWK depends on the cut points a lot, so a fixed rule is needed to compare models fairly.
- The tuned score was misleading for test: final models trained on all train data produce more spread-out test
  predictions than the OOF predictions, so OOF-tuned cuts put too many test pets in classes 1 and 4.
  Train-share cuts on test (submission 06) beat OOF-tuned cuts (submission 05) by +0.0045 LB with the same model.
  From then on dist was the main CV number.

## Leaderboard behaviour
- LB is about CV + 0.17..0.18 (e.g. 07: dist CV 0.622 -> LB 0.8054). The reason is unknown; train and test look alike on
  every statistic except photo count (see `02_data_audit.md`). It could be a smaller or cleaner public subset.
- Noise: versions 10/11/12/15 have practically the same test predictions (correlation >= 0.994, 93-97% equal classes) and
  scored 0.8060..0.8102. Submissions that change many predictions (like 13) can move more than that.
- Practical rule we used: a change is worth submitting only if dist or nested CV improves clearly, about 0.003 or more.

## Where in code
- `_work/feats.py`: `folds`, `fit_cuts`, `nested`, `score`, `tocl`.
- `_work/blend2.py`, `_work/nbr.py`, `_work/stack.py`: `dist_qwk`.
- Notebook section 5 (cells printing "OOF QWK ...").

## Alternatives worth checking
- Repeated CV (several fold seeds) to measure the noise of each CV number; we never measured it directly.
- Group-aware CV by near-duplicate clusters (same template/photos). We kept random folds because test pets also have
  near-duplicates in train, so random folds mimic the test situation.
