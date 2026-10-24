# 10. Negative results (did not help), with numbers

Reference points: LightGBM regression on base features tuned 0.6088 / nested 0.6062; 07-style blend dist 0.6234 (script);
stack with 10 members dist 0.6258 / tuned 0.6315 / nested 0.6292 (script).

| idea | result | reference |
|---|---|---|
| pseudo-labels from test (soft, weighted, confident hard) | 0.6053-0.6075 tuned | 0.6079 (same 2-seed setup) |
| noisy-label drop / down-weight (inner OOF errors, 5-20%) | dist 0.588-0.592 | 0.595 (same 1-seed setup) |
| test-likeness sample weights (previous session) | worse | |
| twin labels as LightGBM features | -0.002..-0.003 | |
| drop training pets with conflicting twins | ~0 | |
| flip TTA embeddings | 0.6022 | 0.6088 |
| DINOv2 large add / replace | 0.6084 / 0.6047 | 0.6088 |
| so400m 384 in LightGBM replace / add | 0.6067 / 0.6088 | 0.6088 |
| PCA 16 / 64 instead of 32 | 0.5988 / 0.6032 | 0.6088 |
| text regex features (age, litter, keywords) | 0.6075 | 0.6088 |
| zero-shot prompt similarities (32 prompts) | 0.6043 | 0.6088 |
| photo file metadata (size, height, bytes per pixel) | 0.6062 | 0.6088 |
| YOLO cat/dog/person counts, animal box size, brightness, sharpness, colourfulness | 0.6063 | 0.6088 |
| group-size counts of similar listings | 0.6060 | 0.6088 |
| no photo count (shift fix), version 14 | LB 0.8068 | 0.8102 |
| e5-large / Qwen3-4B text embeddings | no gain / Ridge 0.34 | |
| CatBoost, MLP, kNN, logistic regression as blend members | small or zero weight | |
| MetaCLIP 2 huge / AIMv2 in members | stack 0.6249/0.6308/0.6297 | 0.6258/0.6315/0.6292 |
| per-photo RBF (Nystroem + Ridge) | 0.589 | per-photo Ridge 0.587 |
| fine-tuned SigLIP base / so400m in blend | no reliable gain | |
| attention over photos (MIL) | 0.571 alone; stack 0.6257/0.6304/0.6249 | 0.6258/0.6315/0.6292 |
| residual propagation over neighbours | dist 0.6210 | 0.6252 |
| label propagation (multi-hop, test-test links) | dist 0.6252 | 0.6247 |
| level-2 huber / multiclass / lambdarank / rank_xendcg | 0.6257 / 0.6247 / 0.594 / 0.557 | 0.6258 |
| level-2 with original features | dist 0.6197 | 0.6263 |
| averaging stacks 10+11+12 | no change | |
| class shares 21/25/19/35 (version 09) | LB 0.8020 | 0.8054 |
| per-species cuts (version 13) | LB 0.7901 | 0.8060 |
| fold-averaged test predictions (version 15) | LB 0.8101 | 0.8102 |
| cumulative ordinal LightGBM, 20% blend with version 11 | dist CV 0.6244 vs 0.6192; public LB 0.8101 | 0.8102 |
| perceptual-hash photo twins (distance <=4), 25% label blend | dist CV 0.6191 vs 0.6192; 108 test pets matched | 0.6192 CV |

Main lesson: once several strong frozen encoders, a kernel model on full vectors and the near-duplicate neighbour signal are
in, every further idea moved CV by less than about 0.003, which is below what the public LB can resolve.
