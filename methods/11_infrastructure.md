# 11. Infrastructure and reproduction

## Layout (inside `final-project-petfinder/`)
- `petfinder_age_course.ipynb` – current course-scope notebook for the best scored age blend (LB 0.8116); `submission.csv` – its submitted output, verified byte-for-byte against `_work/cand/age_proxy30_submission.csv`. `petfinder_final.ipynb` remains the earlier version-11 notebook (LB 0.8102).
- `petfinder_age_fresh.ipynb` – the same training code written into `solution.py` and run in a fresh Python process after package pinning, to check Kaggle reproducibility despite its preloaded NumPy.
- `build_notebook.py` – builds any version: `../.venv/bin/python build_notebook.py --version 11 --out x.ipynb`.
  `VERSIONS` dict at the top holds each version's config (encoders, seeds, flags such as `blend`, `ensemble`, `stack`,
  `dist_cuts`, `nbr`, `pp_agg`, `members10`, `species_cuts`, `no_nphotos`, `fold_test`, `n_folds`).
  Version 02 was the original notebook (copied as is).
- `submissions/NN_*_LBx/` – executed notebook, `submission.csv`, submitted copy `*_submitted.csv`, and from version 08
  on `test_raw.npy` (scores before cuts) and `oof.npy`. `submissions/README.md` – table of all submissions.
- `_work/` – experiment scripts and caches:
  - `feats.py` (shared data, folds, features, metrics, twins), `models.py` (first-level models -> `cache/preds/*.npz`),
    `blend2.py`, `stack.py`, `nbr.py`, `lp.py`, `perphoto.py`, `pp_agg.py`, `mil.py`, `noisy.py`, `detect.py`, ...
  - `cache/*.npy` – pet-level embeddings (rows: 6,431 train then 1,891 test), `*_allphotos.npy` + `allphotos_pet.npy`
    – per-photo vectors (37,831 rows, pet index per row), `cache/preds/*.npz` – OOF and test predictions.
  - `kaggle/*` – Kaggle job scripts with `kernel-metadata.json`; `kaggle_out/*` – their downloaded outputs.

## Running a version locally
For the current age notebook, run `../.venv/bin/python _work/check_age_course_notebook.py` from the project root. It executes every code cell with the local embedding cache, writes `_work/notebook_age_check/submission.csv`, and asserts that all 1,891 labels match the scored 0.8116 CSV.

For an earlier numbered version:
```
cd submissions/<folder>
ln -s ../../data data
EMB_CACHE=<abs path>/_work/cache ../../../.venv/bin/jupyter nbconvert --to notebook --execute --inplace petfinder_NN.ipynb
rm data
```
With the cache, a stacking version runs in about 45 minutes on an M2 Max (CPU-bound). Without the cache the notebook
computes every embedding itself (needs a GPU; SigLIP giant on all photos is the slowest part).

## Kaggle
- CLI `~/.local/bin/kaggle` (token in `~/.kaggle/access_token`), competition `deep-learning-for-computer-vision-and-nlp-2026-10`.
- Private kernels used for GPU work: petfinder-04 (full run of version 04), petfinder-ft-siglip, petfinder-ft-so400m,
  petfinder-emb, petfinder-emb2, petfinder-emb3. At most 2 GPU sessions at once.

## Gotchas
- macOS: running PyTorch after LightGBM in one process deadlocks or segfaults (OpenMP); import LightGBM lazily.
- LightGBM rejects feature names with commas; level-2 columns are renamed to ASCII (`L2_NAMES`).
- Embeddings computed on CUDA vs MPS differ slightly, so a from-scratch Kaggle run gives a slightly different CSV
  (version 04: 92% equal classes, CV 0.607 vs 0.611).
