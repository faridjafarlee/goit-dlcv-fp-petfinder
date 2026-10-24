# 02. Data audit: duplicates, twins, train/test shift

## What was checked
1. Exact duplicate photos (MD5 of the file bytes), grouped into pets that share any identical photo file.
2. Exact duplicate descriptions (lower-cased, whitespace-normalised, at least 40 characters).
3. Near-duplicates: character TF-IDF similarity of descriptions, per-photo SigLIP so400m similarity.
4. Train/test shift: adversarial validation, KS tests on simple statistics.
5. Image sanity: broken files (none), EXIF orientation (all standard), non-RGB images (22, converted to RGB).

## Findings
- Exact photo twins: 196 groups, 435 pets. 95 test pets share an identical photo with a train pet.
- Exact description twins: 130 groups. 77 test pets share a description with a train pet.
- Any exact twin (photo or description): 158 test pets (8.4%); in train CV 6.3% of pets have a twin in other folds.
- Label agreement among train twins (leave-one-out mean of the twins' labels): photo twins QWK 0.73 (accuracy 0.71),
  description twins QWK 0.86 (accuracy 0.83). Much better than the model (QWK ~0.62), but the model already learns
  most of it because identical photos give identical embeddings.
- Near-duplicate descriptions (char_wb 3-5 TF-IDF cosine), label agreement with the most similar train pet:
  - >= 0.9: 8.6% of train pets, QWK 0.773; >= 0.8: 11.3%, 0.743; >= 0.7: 14.5%, 0.674; >= 0.5: 21.8%, 0.582.
  These are most likely listings written from the same volunteer's template.
- Near-duplicate photos (max similarity over photo pairs, so400m): >= 0.99: 4.0%, QWK 0.724; >= 0.95: 8.9%, 0.682;
  >= 0.93: 28%, 0.508 (too loose).
- Shift: adversarial validation AUC 0.54, driven by photo count. Test has more photos (11% single-photo pets vs 17.5%).
  KS tests: photo count differs (p ~ 0); photo size, sharpness, brightness, colourfulness, cat/dog/person counts,
  description length all match (p > 0.1).
- No contact data in descriptions (phones/emails scrubbed: 3 pets of 8,322), so a rescuer ID cannot be rebuilt that way.

## Why it mattered
- The twin/neighbour signal became the main "new information" beyond embeddings (see `06_neighbours_and_twins.md`).
- Removing photo count to fight the shift did not help (submission 14, LB 0.8068 vs 0.8102).

## Code
- `_work/audit_dups.py`, `_work/near_check.py`, `_work/nbr.py`, `_work/detect.py`, notebook section "Двійники в даних"
  (versions 04-06).

## Alternatives
- Perceptual hashing (pHash) of photos to catch re-encoded duplicates with a different tool than embeddings.
- Clustering listings into pseudo-rescuers (text style + photo backgrounds) and using cluster-level target encoding.
  Our label-free "group size" counts did not help, supervised neighbour labels did a little.
