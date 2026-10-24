# 12. Open ideas (not tried or not finished)

Gap to the public top score: 0.8265 - 0.8116 = 0.0149 LB. Our CV plateau: ~0.627 tuned / ~0.626 train-share cuts.

## Not tried
- DINOv3 (needs licence acceptance on Hugging Face), Perception Encoder (PE-Core, open_clip), InternViT.
- Prompted LLM / vision-LLM attribute extraction was considered, but is outside the user's course-method boundary; do not use it for a submission.
- Translation of Malay/Chinese descriptions before embedding.
- Metric learning of listing similarity (pairs of pets with the same / different label) to replace fixed thresholds.
- LoRA fine-tuning of a large CLIP-like model with all photos per pet and an ordinal loss.
- Repeated CV with several fold seeds, bagging the whole pipeline across fold splits.
- Expected-QWK-optimal cut points from class probabilities.

## Finished after writing
- Version 16 (version 11 with 10-fold first level): CV tuned 0.6275 / dist 0.6191 vs 0.6278 / 0.6192, not submitted.

## Course-scope experiments completed on 2026-10-04
- Private Kaggle kernel `faridjafarlee/petfinder-ft-multitask` added an auxiliary photo-age ordered head during SigLIP 2 fine-tuning. Its high-confidence age labels came only from supplied descriptions. Five folds completed, but its best substantive blend was worse than the current submission; see audit entry 57.
- Private Kaggle kernel `faridjafarlee/petfinder-pe-core-emb` extracted frozen first-three-photo features with Meta's [PE-Core-L14-336](https://huggingface.co/timm/vit_pe_core_large_patch14_336.fb) via timm. Direct heads and age-feature augmentation did not improve the current blend; see audit entry 58.

## Completed after writing
- Frozen 512px SigLIP 2 and YOLO-cropped 384px embeddings were extracted from only the provided competition photos. All 8,322 rows passed pet-order and missing-photo checks. Neither improved the submitted age blend; details and scores are in `13_alternative_audit.md` entries 22–23.
- Multilingual DistilBERT was fine-tuned on the supplied descriptions with five adoption folds. It did not improve the age blend under nested weight selection; see `13_alternative_audit.md` entry 29.
- Frozen ConvNeXt V2 large features were extracted and row-order verified but did not improve the age blend; see `13_alternative_audit.md` entry 31.
- Frozen animal-identification CLIP features were extracted and row-order verified; neither their direct adoption heads nor identity-focused nearest-neighbor retrieval improved the current candidate enough to submit; see `13_alternative_audit.md` entry 39. Model source: https://huggingface.co/AvitoTech/CLIP-ViT-base-for-animal-identification.
- Frozen Oxford Pets breed-classifier features were extracted from the supplied photos. Despite a raw association between breed confidence and adoption speed, probability, confidence, species, and CLS feature variants all lowered five-fold adoption QWK; see `13_alternative_audit.md` entry 40.
- Fine-tuned SigLIP 2 with an ordered cumulative-logit adoption head completed on Kaggle, but did not improve the current blend; see `13_alternative_audit.md` entry 44.
- Frozen EVA-CLIP features completed on Kaggle; direct Ridge/SVR adoption heads and age-LightGBM augmentation did not improve the current blend; see `13_alternative_audit.md` entry 51.

## Deliberately not used
- The public original PetFinder.my competition data: it contains these same pets and their labels, so any use would leak
  test answers.
- Leaderboard probing to reverse-engineer labels.

## Questions for an alternative approach
- Is there a modelling family that would add information beyond frozen CLIP/SigLIP features plus near-duplicate
  neighbours? Every model type we tried collapsed onto the same ~0.627 CV.
- Can the CV-to-LB gap (+0.17) be explained? Train and test look identical except for photo count.
