# 05. First-level models

All models use the same 5 folds and produce OOF predictions on the 1..4 scale plus test predictions.
"Base features" = photo count, description word count, TF-IDF SVD 32, MiniLM PCA 32, and PCA 32 of
EfficientNet-B0, SigLIP 2 so400m, CLIP ViT-L/14, DINOv2 base (194 columns).

| model | input | CV tuned / nested | used in |
|---|---|---|---|
| LightGBM regression, 3 seeds | base features | 0.609 / 0.606 | all versions |
| LightGBM multiclass, expected class (proba @ [1,2,3,4]), 3 seeds | base features | 0.612 / 0.607 | 05+ |
| LightGBM multiclass, "all photo models" | base + PCA32 of SigLIP 2 base and DINOv2 large | 0.612 / 0.611 | 07, 10+ |
| LightGBM extra_trees regression | base features | 0.612 / 0.608 | 12 |
| CatBoost regression (depth 6) | base features | 0.599 / 0.595 | 12 |
| Ridge, average over inputs | char TF-IDF, MiniLM and each full photo vector (standardised, alpha = 300*dim/768) | 0.605 / 0.597 (8 photo encoders) | 05+ |
| SVR RBF (C=1, eps=0.3), average over inputs | PCA 256 of each full photo vector | 0.609 / 0.606 (8 encoders) | 07+ |
| per-photo Ridge (alpha 1e4), mean per pet | every photo vector labelled with its pet's label | 0.587-0.590 per encoder, 0.593 avg of 4 | 11 (as stats), 12 |
| kNN (k=50, exp(10*cos) weights) | so400m+CLIP vectors | 0.546 | 12 |
| multinomial logistic regression, expected class | PCA128 of 4 photo vectors | 0.576 | 12 |
| MLP on PCA features / on full embeddings | | 0.557 / 0.554 | dropped |

LightGBM settings: lr 0.02, 31 leaves, min_child_samples 30, feature_fraction 0.7, bagging 0.8, lambda_l2 1,
early stopping 200 on the validation fold; final model on all train with mean best iteration x 1.1.

## Why these models
- LightGBM is the strongest on the mixed compressed features; multiclass expected value was slightly better than
  regression and errs differently.
- Linear and kernel models on the full, uncompressed vectors use information PCA throws away; SVR beat Ridge on every
  photo encoder (+0.01..0.02).
- Averaging one Ridge/SVR per encoder instead of one model on the concatenation: simpler, robust, each encoder gets its
  own regularisation.
- Per-photo training gives ~5x more training rows and lets each photo speak; it beat mean-vector Ridge by ~0.015.
- kNN, logistic regression, CatBoost and extra-trees were added only for variety in the stack.

## Tried, no gain
- Hyperparameter variants of LightGBM (smaller trees, stronger regularisation): same CV.
- Photo-count removal (version 14), text regex, zero-shot prompts, photo metadata, YOLO object counts and photo quality
  features: all equal or worse (see `10_negative_results.md`).

## Code
- `_work/models.py` (lgb, lgbmc, cat, ridge, svr, mlp, logreg, knn), `_work/perphoto.py`, `_work/pp_rbf.py`;
  notebook cells in sections 5 (members) of versions 05-16.
