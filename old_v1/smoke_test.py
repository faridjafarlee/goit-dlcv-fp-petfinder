"""Smoke test of petfinder_final_template.ipynb on SYNTHETIC competition-like data (random photos).

Nothing here touches Kaggle. The script creates a temp folder with fake CSVs and random JPEGs, inserts a
CONFIG-override cell right after the notebook's CONFIG cell, executes the notebook with nbclient and checks
the results. The executed copies stay in the temp folder - the committed notebook is not modified.

Positive runs
  A  original-like schema: train/train.csv, test/test.csv, test/sample_submission.csv (+ a decoy breed_labels.csv),
     RescuerID and tabular columns, target 0-4, photos train_images/<PetID>-<n>.jpg (some pets without photos,
     corrupt files), *_sentiment / *_metadata JSON. EfficientNet-B0 (torchvision, ImageNet weights), LightGBM,
     Optuna (3 trials), text embeddings from a tiny random BERT. Executed twice with fresh output folders:
     the submission SHA-256 must be identical.
  B  reduced schema: only PetID / Description / AdoptionSpeed (1-4), test file named valid.csv, shuffled
     sample_submission, photos in per-pet folders images/<PetID>/photo_<k>.jpg; Hugging Face CLIP image encoder
     (tiny random test model), sklearn HistGradientBoosting fallback, 'distribution' rounding, 'full' prediction.
Negative runs (the notebook must stop with a clear message)
  C  competition data without any photos       -> "Не знайшов жодного фото"
  D  test file without the Description column  -> "немає колонки"

Usage:  ../.venv/bin/python smoke_test.py [--workdir DIR]
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError
from PIL import Image

HERE = Path(__file__).resolve().parent
NB_PATH = HERE / 'petfinder_final_template.ipynb'
MAX_IMG = 3
WORDS = ('friendly playful cute kitten puppy vaccinated healthy adopt loving home active gentle rescued shy calm '
         'energetic dewormed spayed neutered found street owner family please call contact good boy girl').split()


def rand_text(rng, label):
    words = list(rng.choice(WORDS, size=int(rng.integers(0, 40))))
    if label <= 1:
        words += ['cute', 'playful']
    return (' '.join(words).capitalize() + '.') if words else ''


def save_photo(path, rng, label, corrupt=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if corrupt:
        path.write_bytes(b'this is not a jpeg')
        return
    w, h = int(rng.integers(120, 320)), int(rng.integers(120, 320))
    arr = rng.integers(0, 255, size=(h, w, 3)).astype(np.float32)
    arr[..., label % 3] = arr[..., label % 3] * 0.5 + 120          # a little class-dependent tint
    Image.fromarray(arr.clip(0, 255).astype(np.uint8)).save(path, quality=85)


def make_ids(rng, n):
    ids = set()
    while len(ids) < n:
        ids.add(f'{int(rng.integers(16 ** 8, 16 ** 9)):x}')
    return sorted(ids)


def make_variant_a(root, rng):
    n_train, n_test = 30, 10
    ids = make_ids(rng, n_train + n_test)
    y = np.repeat([0, 1, 2, 3, 4], 6)
    rng.shuffle(y)
    labels = list(y) + [int(rng.integers(0, 5)) for _ in range(n_test)]
    n = n_train + n_test
    df = pd.DataFrame({
        'Type': rng.integers(1, 3, n), 'Name': rng.choice(['Milo', 'Luna', 'No Name', ''], n), 'Age': rng.integers(1, 60, n),
        'Breed1': rng.integers(1, 300, n), 'Gender': rng.integers(1, 4, n), 'Color1': rng.integers(1, 8, n),
        'Quantity': rng.integers(1, 4, n), 'Fee': rng.choice([0, 50, 100], n), 'State': rng.choice([41326, 41401, 41327], n),
        'RescuerID': rng.choice([f'resc{i}' for i in range(8)], n), 'VideoAmt': 0,
        'Description': [rand_text(rng, l) for l in labels], 'PetID': ids})
    df.loc[3, 'Description'] = np.nan
    expected_valid = {}
    n_photos = rng.integers(0, 4, n)
    for i, pid in enumerate(ids):
        folder = root / ('train_images' if i < n_train else 'test_images')
        flags = []
        for k in range(1, int(n_photos[i]) + 1):
            corrupt = (i == 5 and k == 1)                            # pet 5: first photo corrupt
            save_photo(folder / f'{pid}-{k}.jpg', rng, labels[i], corrupt=corrupt)
            flags.append(not corrupt)
        if i == 7:                                                   # pet 7: its only photo is corrupt
            for f in folder.glob(f'{pid}-*.jpg'):
                f.unlink()
            save_photo(folder / f'{pid}-1.jpg', rng, labels[i], corrupt=True)
            flags = [False]
        expected_valid[pid] = sum(flags[:MAX_IMG])
    df['PhotoAmt'] = [len(list((root / ('train_images' if i < n_train else 'test_images')).glob(f'{p}-*.jpg')))
                      for i, p in enumerate(ids)]
    (root / 'train').mkdir()
    (root / 'test').mkdir()
    train = df.iloc[:n_train].assign(AdoptionSpeed=y)
    test = df.iloc[n_train:]
    train.to_csv(root / 'train' / 'train.csv', index=False)
    test.to_csv(root / 'test' / 'test.csv', index=False)
    pd.DataFrame({'PetID': test.PetID, 'AdoptionSpeed': 0}).to_csv(root / 'test' / 'sample_submission.csv', index=False)
    pd.DataFrame({'BreedID': [1, 2], 'Type': [1, 1], 'BreedName': ['Affenpinscher', 'Afghan Hound']}).to_csv(
        root / 'breed_labels.csv', index=False)
    for i, pid in enumerate(ids[:20]):
        folder = root / ('train_sentiment' if i < n_train else 'test_sentiment')
        folder.mkdir(exist_ok=True)
        (folder / f'{pid}.json').write_text(json.dumps({
            'documentSentiment': {'magnitude': float(rng.random() * 3), 'score': float(rng.random() * 2 - 1)},
            'sentences': [{}] * int(rng.integers(1, 5)), 'entities': [], 'language': 'en'}))
    for i, pid in enumerate(ids[:15]):
        folder = root / 'train_metadata'
        folder.mkdir(exist_ok=True)
        (folder / f'{pid}-1.json').write_text(json.dumps({
            'labelAnnotations': [{'description': 'cat', 'score': float(rng.random())}],
            'imagePropertiesAnnotation': {'dominantColors': {'colors': [
                {'color': {'red': 120, 'green': 80, 'blue': 60}, 'score': 0.4, 'pixelFraction': 0.2}]}},
            'cropHintsAnnotation': {'cropHints': [{'confidence': 0.8, 'importanceFraction': 1.0}]}}))
    return {'train_ids': ids[:n_train], 'test_ids': ids[n_train:], 'classes': [0, 1, 2, 3, 4],
            'valid_photos': expected_valid, 'sample_order': list(test.PetID), 'sub_name': 'submission.csv'}


def make_variant_b(root, rng):
    n_train, n_test = 30, 10
    ids = make_ids(rng, n_train + n_test)
    y = np.array([1] * 8 + [2] * 8 + [3] * 7 + [4] * 7)
    rng.shuffle(y)
    labels = list(y) + [int(rng.integers(1, 5)) for _ in range(n_test)]
    desc = [rand_text(rng, l) for l in labels]
    pd.DataFrame({'PetID': ids[:n_train], 'Description': desc[:n_train], 'AdoptionSpeed': y}).to_csv(root / 'train.csv', index=False)
    pd.DataFrame({'PetID': ids[n_train:], 'Description': desc[n_train:]}).to_csv(root / 'valid.csv', index=False)
    order = list(rng.permutation(ids[n_train:]))
    pd.DataFrame({'PetID': order, 'AdoptionSpeed': 1}).to_csv(root / 'sample_submission.csv', index=False)
    expected_valid = {}
    for i, pid in enumerate(ids):
        k_max = int(rng.integers(0, 5))
        for k in range(1, k_max + 1):
            save_photo(root / 'images' / pid / f'photo_{k}.jpg', rng, labels[i])
        expected_valid[pid] = min(k_max, MAX_IMG)
    return {'train_ids': ids[:n_train], 'test_ids': ids[n_train:], 'classes': [1, 2, 3, 4],
            'valid_photos': expected_valid, 'sample_order': [str(x) for x in order], 'sub_name': 'submission.csv'}


def execute(overrides, run_dir):
    nb = nbformat.read(NB_PATH, as_version=4)
    idx = next(i for i, c in enumerate(nb.cells) if 'config' in c.metadata.get('tags', []))
    src = '# SMOKE TEST: CONFIG overrides\n' + '\n'.join(f'{k} = {v!r}' for k, v in overrides.items())
    nb.cells.insert(idx + 1, nbformat.v4.new_code_cell(src))
    run_dir.mkdir(parents=True, exist_ok=True)
    client = NotebookClient(nb, timeout=900, kernel_name='python3', resources={'metadata': {'path': str(run_dir)}})
    try:
        client.execute()
        return nb, None
    except CellExecutionError as e:
        return nb, str(e)
    finally:
        nbformat.write(nb, run_dir / 'executed.ipynb')


def check_positive(out_dir, exp, cv_scheme_contains):
    sub_path = out_dir / exp['sub_name']
    header = sub_path.read_text().splitlines()[0]
    assert header == 'PetID,AdoptionSpeed', header
    sub = pd.read_csv(sub_path, dtype={'PetID': str})
    assert len(sub) == len(exp['test_ids']) and set(sub.PetID) == set(exp['test_ids']), 'wrong PetIDs'
    assert list(sub.PetID) == exp['sample_order'], 'row order differs from sample_submission'
    assert set(sub.AdoptionSpeed) <= set(exp['classes']), sub.AdoptionSpeed.unique()
    summary = json.loads((out_dir / 'run_summary.json').read_text())
    assert summary['mode'] == 'competition' and summary['classes'] == exp['classes'], summary['classes']
    assert cv_scheme_contains in summary['cv_scheme'], summary['cv_scheme']
    groups = json.loads((out_dir / 'feature_groups.json').read_text())
    assert 'image' in groups, list(groups)
    X = pd.read_pickle(out_dir / 'features.pkl')     # trusted: written by the notebook in this same run
    pos = {pid: i for i, pid in enumerate(exp['train_ids'] + exp['test_ids'])}
    pca_cols = [c for c in groups['image'] if c.startswith('img_pca')]
    for pid, n_valid in exp['valid_photos'].items():
        row = X.iloc[pos[pid]]
        if n_valid == 0:
            assert row['img_missing'] == 1 and row[pca_cols].isna().all(), (pid, 'expected missing image features')
        else:
            assert row['img_missing'] == 0 and row[pca_cols].notna().all() and row['img_n_used'] == n_valid, (
                pid, n_valid, row['img_n_used'])
    return {'sha256': hashlib.sha256(sub_path.read_bytes()).hexdigest(), 'groups': {g: len(c) for g, c in groups.items()},
            'n_img_pca': len(pca_cols), 'qwk_nested': summary['cv_metrics']['qwk_nested_thresholds'],
            'pets_missing_img': int(sum(v == 0 for v in exp['valid_photos'].values())),
            'runtime_sec': summary['runtime_sec'], 'image_backbone': summary['image_backbone'], 'model': summary['model']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workdir', default=None, help='where to put synthetic data and executed copies (default: new temp dir)')
    args = ap.parse_args()
    work = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix='petfinder_smoke_'))
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    try:
        import certifi
        os.environ.setdefault('SSL_CERT_FILE', certifi.where())
    except ImportError:
        pass
    rng = np.random.default_rng(0)
    data_a, data_b = work / 'data_a', work / 'data_b'
    data_a.mkdir()
    data_b.mkdir()
    exp_a = make_variant_a(data_a, rng)
    exp_b = make_variant_b(data_b, rng)
    n_jpg = len(list(data_a.rglob('*.jpg'))) + len(list(data_b.rglob('*.jpg')))
    print(f'Synthetic data in {work}: {n_jpg} JPEG files, variant A 30+10 pets, variant B 30+10 pets')

    common = dict(ALLOW_PROXY=False, N_FOLDS=3, MAX_IMAGES_PER_PET=MAX_IMG, IMG_BATCH_SIZE=16, RUN_ABLATION=True)
    ovr_a = dict(common, DATA_ROOT=str(data_a), MODEL_SEEDS=[42, 7], USE_OPTUNA=True, OPTUNA_TRIALS=3, EARLY_STOPPING=30,
                 LGB_PARAMS=dict(objective='regression', learning_rate=0.1, n_estimators=300, num_leaves=7,
                                 min_child_samples=3, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                 min_data_per_group=3, cat_smooth=1),
                 TEXT_MODEL='hf-internal-testing/tiny-random-BertModel', TEXT_BATCH_SIZE=16)
    ovr_b = dict(common, DATA_ROOT=str(data_b), MODEL_SEEDS=[42], MODEL_LIB='sklearn', USE_TEXT_EMB=False,
                 HGB_PARAMS=dict(learning_rate=0.1, max_iter=100, max_leaf_nodes=7, min_samples_leaf=3, early_stopping=False),
                 IMAGE_BACKBONE='hf:hf-internal-testing/tiny-random-CLIPModel', ROUNDING='distribution',
                 FINAL_PREDICTION='full')

    results, ok = [], True

    def positive(name, ovr, exp, cv_scheme):
        nonlocal ok
        t0 = time.time()
        out = work / f'out_{name}'
        _, err = execute(dict(ovr, OUTPUT_DIR=str(out)), work / f'run_{name}')
        if err:
            ok = False
            results.append((name, 'FAIL', err[-1500:]))
            return None
        try:
            info = check_positive(out, exp, cv_scheme)
        except AssertionError as e:
            ok = False
            results.append((name, 'FAIL', f'check failed: {e}'))
            return None
        results.append((name, 'OK', f'{time.time() - t0:.0f} s; {info}'))
        return info

    def negative(name, ovr, must_contain):
        nonlocal ok
        _, err = execute(dict(ovr, OUTPUT_DIR=str(work / f'out_{name}')), work / f'run_{name}')
        if err and must_contain in err:
            results.append((name, 'OK', f'stopped with the expected message ("{must_contain}")'))
        else:
            ok = False
            results.append((name, 'FAIL', 'no error' if not err else err[-800:]))

    a1 = positive('A_run1', ovr_a, exp_a, 'RescuerID')
    a2 = positive('A_run2', ovr_a, exp_a, 'RescuerID')
    if a1 and a2:
        same = a1['sha256'] == a2['sha256']
        ok &= same
        results.append(('A_reproducible', 'OK' if same else 'FAIL', f'sha256 run1 == run2: {same}'))
    positive('B', ovr_b, exp_b, 'StratifiedKFold')

    data_c = work / 'data_c'
    data_c.mkdir()
    shutil.copy(data_b / 'train.csv', data_c / 'train.csv')
    shutil.copy(data_b / 'valid.csv', data_c / 'test.csv')
    negative('C_no_images', dict(common, DATA_ROOT=str(data_c), USE_TEXT_EMB=False), 'Не знайшов жодного фото')
    data_d = work / 'data_d'
    shutil.copytree(data_b, data_d)
    pd.read_csv(data_d / 'valid.csv').drop(columns=['Description']).to_csv(data_d / 'valid.csv', index=False)
    negative('D_no_text_in_test', dict(common, DATA_ROOT=str(data_d), USE_TEXT_EMB=False), 'немає колонки')

    lines = [f'Smoke test {time.strftime("%Y-%m-%d %H:%M")}  (workdir: temp folder, {n_jpg} synthetic JPEGs)']
    for name, status, msg in results:
        lines.append(f'[{status}] {name}: {msg}')
    lines.append('ALL OK' if ok else 'SOME CHECKS FAILED')
    report = '\n'.join(lines)
    print(report)
    (HERE / 'smoke_test_last_run.txt').write_text(report + '\n')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
