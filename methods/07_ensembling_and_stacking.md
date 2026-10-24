# 07. Ensembling and stacking

## Weighted blends (versions 05-09)
- 05/06: 0.5 x LGB regression + 0.3 x LGB multiclass + 0.2 x Ridge. Weights from a grid on OOF.
  CV tuned 0.617 vs 0.609 for LGB alone. LB 0.7902 (tuned cuts) / 0.7947 (train-share cuts).
- 07: 0.28 LGB reg + 0.10 LGB multiclass + 0.24 Ridge + 0.24 SVR + 0.14 LGB multiclass on all photo models.
  Weights by hill climbing on the dist metric (`_work/blend2.py`). LB 0.8054.
- Why: models of different kinds (trees on compressed features, linear/kernel on full vectors) make different errors;
  the blend was the biggest single jump after adding encoders.

## Stacking (versions 10-16)
- Level 2: small LightGBM regression (lr 0.01, 7 leaves, min_child 50, feature_fraction 0.8, lambda_l2 5), trained with
  the same folds on: OOF predictions of the members + neighbour features (`06_*`) + photo count.
  Version 11 also adds per-photo prediction statistics (mean/max/min/std/first photo) of per-photo Ridge for 4 encoders.
- Test: members' test predictions -> level-2 model trained on all OOF rows.
- Why: lets the second model learn when to trust neighbours and which member to trust per pet; on LB it beat the fixed
  blend by ~+0.005 although dist CV was the same (0.622).

## Variants tried
| variant | CV (dist / tuned / nested) | LB |
|---|---|---|
| 10: 5 members | 0.622 / 0.627 / – | 0.8100 |
| 11: + per-photo stats | 0.619 / 0.628 / – | 0.8102 |
| 12: 10 members, 5 level-2 seeds | 0.622 / 0.627 / – | 0.8060 |
| 15: fold-averaged test predictions for level 2 | same as 11 | 0.8101 |
| 16: 10-fold first level | 0.619 / 0.6275 / – | not submitted |
| richer neighbour features | 0.625 / 0.630 / 0.624 (script) | not submitted |
| level 2 objective huber / multiclass | 0.626 / 0.625 (script) | not submitted |
| level 2 lambdarank / rank_xendcg | 0.594 / 0.557 | not submitted |
| level 2 with the original 193 features | 0.620 | not submitted |
| averaging stacks 10+11+12 | no change (test preds correlated 0.994+) | |

## Code
- `_work/blend2.py`, `_work/stack.py`; notebook versions 05-16 (`STACK`, `MEMBERS10`, `PP_AGG`, `FOLD_TEST` flags).

## Alternatives
- Level 2 as a constrained linear model (non-negative weights) plus a small neighbour correction, which may be more stable.
- Optimise the blend directly for QWK with train-share cuts (rank-based) using differentiable surrogates.
