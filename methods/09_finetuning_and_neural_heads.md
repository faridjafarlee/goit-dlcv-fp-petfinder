# 09. Fine-tuning and neural heads

## Fine-tuning image encoders (Kaggle GPU)
- Script: `_work/kaggle/petfinder-ft/ft.py` (SigLIP 2 base) and `_work/kaggle/petfinder-ft-so400m/ft.py` (so400m).
- Setup: vision tower + dropout + linear head, MSE on standardised target, AdamW, layer-wise LR decay (0.8 / 0.85),
  head LR 1e-3, backbone LR 4e-5 / 2e-5, cosine schedule with warm-up, fp16, 5 / 4 epochs, batch 32 / 16 (so400m with
  gradient checkpointing). Each epoch a random photo of the first 3 per pet; prediction = mean over the first 3 photos and
  their mirror images. Same 5 folds; best epoch by validation RMSE.
- Results: SigLIP base OOF RMSE 0.907, tuned QWK 0.571; so400m RMSE 0.889, tuned QWK 0.591 (about 3 hours on T4).
- In blends/stacks: no reliable gain (best weight ~0.07, CV unchanged).
- Why it did not help: frozen so400m + SVR already reaches 0.592; 6.4k noisy labels are not enough for fine-tuning to
  learn something new; the fine-tuned model mostly repeats what the frozen features know.

## MLP heads
- MLP on the PCA features of the LightGBM set: 0.557.
- MLP on full embeddings of 8 encoders (PCA 256 each, 2 hidden layers, dropout 0.5, 5 seeds): 0.554.
- Overfits; tree and kernel models are better on this size of data.

## Attention over all photos (multiple-instance learning)
- `_work/mil.py`: per-photo vectors of 4 encoders (PCA 128 each), up to 12 photos per pet, flag for the first photo,
  gated attention pooling, small head, smooth L1, 3 seeds.
- Result: 0.571 alone, no gain in the stack.

## Alternatives
- LoRA fine-tuning of a large CLIP/SigLIP with multi-photo input and an ordinal loss, longer training, strong augmentation.
- Joint image+text model (e.g. SigLIP image tower + multilingual text tower with a fusion head).
- Knowledge distillation from the ensemble into a single fine-tuned model.
