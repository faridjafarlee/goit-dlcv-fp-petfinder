# 04. Text features (Description)

## What
- Word TF-IDF (1-2 grams, min_df 3, 20k features, sublinear) -> TruncatedSVD 32 for LightGBM.
- paraphrase-multilingual-MiniLM-L12-v2 mean-pooled embedding -> PCA 32 for LightGBM; full vector for Ridge.
- Character TF-IDF (char 2-6, 300k features) for the Ridge member from version 08 on.
- Description word count as a simple feature.

## Results (text alone, Ridge, tuned QWK)
| representation | QWK |
|---|---|
| word TF-IDF 1-2 | 0.346 |
| char_wb 3-5 TF-IDF | 0.350 |
| char 2-6 TF-IDF | 0.370 (best) |
| MiniLM multilingual | 0.249 |
| multilingual-e5-large | 0.324 |
| Qwen3-Embedding-4B (with task instruction) | 0.340 |

## Why
- Text is a weak signal here (images are ~0.59); the descriptions are short, many are templates.
- Multilingual models because descriptions mix English, Malay and some Chinese.
- Char n-grams catch writing style and templates, which is closer to "who wrote it" than meaning.
- Text adds a little in LightGBM; stronger text embeddings (e5, Qwen) did not raise the ensemble.

## Did not help
- e5-large in LightGBM (0.6047 vs 0.6046 tuned).
- Qwen3-Embedding-4B (Ridge 0.34).
- Regex features: age in months, litter size, 36 keywords (vaccinated, free, urgent, breed names...),
  character statistics (LGB 0.6075 vs 0.6088).

## Code
- Notebook section 4.2; `_work/feats.py` (`build`, `regex_feats`), `_work/qwen_local.py`, `_work/embed_e5.py`.

## Alternatives
- Translate non-English descriptions before embedding.
- Ask an LLM to extract structured facts (age, sex, health, vaccinated, fee, number of pets) per listing.
- Fine-tune a small text model on the target (risky with 6.4k noisy labels).
