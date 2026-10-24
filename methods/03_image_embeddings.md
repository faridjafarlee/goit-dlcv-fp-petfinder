# 03. Image embeddings

## What
Frozen pretrained encoders turn each photo into a vector. Pooling per pet:
- first-3 mean: mean vector of photos 1-3 (photo 1 is the main one). Used by most models.
- all-photo mean: mean over all photos of the pet ("...all" names).
- per-photo: one vector per photo, used by per-photo models (see `05_level1_models.md`).
For LightGBM each pet vector is reduced with PCA to 32 numbers per encoder; Ridge/SVR use the full vector.

## Encoders and single-model results (CV QWK, tuned cuts)
| encoder | Ridge (first-3) | SVR (first-3) | notes |
|---|---|---|---|
| EfficientNet-B0 (ImageNet, 1280-d) | 0.463 | 0.505 | weakest, kept in LGB features |
| SigLIP 2 base 224 | ~0.57 | | replaced by so400m |
| SigLIP 2 so400m 224 (1152-d) | 0.574 | 0.592 | main encoder |
| SigLIP 2 so400m 384 | 0.572 | 0.598 | helps SVR only |
| SigLIP 2 so400m, all photos mean | 0.578 | 0.596 | |
| SigLIP 2 giant-opt 384 (Kaggle GPU) | 0.577 | 0.592 | |
| CLIP ViT-L/14 224 | 0.570 | 0.585 | |
| CLIP ViT-L/14 336 | 0.570 | 0.587 | |
| DINOv2 base (CLS + mean patch) | 0.530 | 0.547 | |
| DINOv2 large | | | no gain in LGB |
| MetaCLIP 2 huge (all-photo mean) | 0.574 | 0.591 | no gain in the stack |
| AIMv2 large (all-photo mean) | 0.559 | 0.563 | weaker |

## Why this way
- Only ~6.4k labelled pets: frozen strong encoders + simple heads beat fine-tuning here (see `09_*`).
- Contrastive image-text models (SigLIP, CLIP) were clearly better than ImageNet CNNs and DINO for this target:
  they encode breed, age, cuteness, photo setting, which matter for adoption.
- Several encoders together help because they err differently (EffNet+SigLIP: 0.506 -> 0.576 CV; + CLIP + DINO: 0.601).
- PCA 32 for LightGBM: 16 and 64 were worse (0.599 and 0.603 vs 0.609).
- Normalised vectors averaged over photos; NaN for 4 pets without photos.

## What did not help
- Horizontal-flip test-time augmentation of the embeddings (LGB 0.602 vs 0.609).
- DINOv2 large added or replacing base (0.608 / 0.605 vs 0.609).
- so400m 384 inside LightGBM (0.607 replacing, 0.609 adding).
- Zero-shot prompt similarities (32 prompts like "a photo of a puppy") as extra features (0.604).

## Code / caches
- `_work/embed.py`, `_work/embed_more.py` (Mac MPS), Kaggle scripts `_work/kaggle/petfinder-emb*/` (CUDA, fp16).
- Cache files `_work/cache/<name>.npy` (pet x dim, rows = train then test) and `<name>_allphotos.npy` + `allphotos_pet.npy`.
- Notebook: `IMAGE_ENCODERS`, `photo_embeddings`, `cached` in section 4.3.

## Alternatives
- DINOv3 (gated on HF, needs licence acceptance), EVA-02 CLIP, Perception Encoder (PE-Core), InternViT.
- Higher resolution SigLIP 2 (512) or multi-crop pooling.
- Learned pooling over photos instead of mean (our attention head was weaker, see `09_*`).
