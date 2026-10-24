# 06. Neighbours and twins (similar listings)

## Idea
Many listings are near-copies: the same volunteer reuses a description template, re-uploads the same photos, or lists a
whole litter. Such pets tend to have similar adoption speed, and embeddings compressed for LightGBM do not see "this is
almost the same listing as pet X". Using the labels of very similar train pets adds information the models lack.

## Versions of the idea
1. Exact twins (version 04-06): same photo file (MD5) or same description. Prediction mixed 50/50 with the mean label of
   twins in the training part. Small gain with strong models (+0.001..0.003 CV).
2. Near-duplicate descriptions (version 07): character TF-IDF (char_wb 3-5) cosine >= 0.7, descriptions >= 40 chars.
   Prediction (after putting the blend back on the label scale) mixed 50/50 with the similarity-weighted mean label of such
   train neighbours. 07 was the first submission above 0.80 (LB 0.8054). CV +0.004..0.005.
3. Photo-pair neighbours (version 08): for each pair of pets the maximum cosine over all photo pairs (so400m per photo),
   threshold 0.95, added to the description neighbours. CV +0.001, LB 0.8003 (noise or no gain).
4. Neighbour features inside stacking (version 10+): for text sim >= 0.5 and photo-mean sim >= 0.9: similarity-weighted
   mean label, best similarity, count, label of the closest train pet. The level-2 model decides how much to trust them.
   This gave the best LB (0.8100 / 0.8102).

## Fold honesty
Neighbours of an OOF row always come from the training part of that fold; for test pets all train pets are used.

## Tried, no gain
- Twin labels as LightGBM features (instead of post-blend): worse.
- Dropping training pets whose twins have different labels: no gain.
- Similarity-dependent mixing weight: 0.6259 vs 0.6252 dist.
- Residual propagation (add neighbours' residual y - prediction): 0.6210 vs 0.6252.
- Label propagation over the full graph (multi-hop, test-test links, labels clamped): 0.6252 vs 0.6247.
- Unsupervised "group size" counts (how many similar listings a pet has): no gain.

## Code
- `_work/nbr.py`, `_work/nbr2.py`, `_work/lp.py`, `_work/resid.py`, `_work/stack.py` (`nfeat`),
  notebook cells "NBR" (07-09) and "STACK" (10+).

## Alternatives
- Learn the similarity (metric learning on pairs with equal/unequal labels) instead of fixed TF-IDF/cosine thresholds.
- Combine text and photo similarity into one learned pair model.
