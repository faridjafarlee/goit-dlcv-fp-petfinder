# 08. Cut points: from scores to classes

## Options tried
1. OOF-tuned cuts (versions 01-05): coordinate search maximising QWK on OOF predictions, applied to test.
2. Train-share cuts (06 onward): on test, cut points are quantiles of the test predictions at the train class shares
   (18.6 / 27.6 / 20.6 / 33.2%), so the predicted class mix equals train's.
3. Shifted shares (09): 21 / 25 / 19 / 35% (more extreme classes).
4. Per-species shares (13): separate quantile cuts for cats and dogs (species from SigLIP zero-shot,
   "a photo of a cat" vs "a photo of a dog"), each at its own train class shares
   (cats 25.8/28.4/15.9/29.9%, dogs 13.1/26.9/24.3/35.7%).

## Results
| rule | CV (dist-style) | LB |
|---|---|---|
| OOF-tuned (05) | – | 0.7902 |
| train shares (06, same model as 05) | 0.610 | 0.7947 |
| shifted 21/25/19/35 (09, same model as 07) | – | 0.8020 vs 0.8054 |
| per species (13, same model as 12) | 0.624 vs 0.622 | 0.7901 vs 0.8060 |

## Why train shares
- Final models trained on all data give more spread-out test scores (std 1.18 vs ~1.08 for OOF on the label scale), so
  OOF-tuned cuts over-assigned classes 1 and 4 on test. Quantile cuts are immune to that scale shift and only depend on
  the ranking.
- Black-box shift estimation of test class shares was too unstable to use (bootstrap ranges 0-47% per class).
- Per-species cuts looked better in CV but clearly worse on LB, so we kept one global rule.

## Code
- Notebook cell after "OOF QWK з підібраними порогами" (`SHARES`, `TEST_CUTS`, `species_classes`).

## Alternatives
- Cuts chosen by expected-QWK maximisation under the predicted class probabilities (e.g. from the multiclass member).
- Hedging between OOF-tuned and train-share cuts.
