"""Builds a version of the final project notebook (PetFinder.my adoption speed, metric QWK).

Every Kaggle submission has its own version: which photo models, how many LightGBM seeds,
and whether the twin step is used. Version 02 is the notebook that was run as is (see
_work/build_notebook_v02_backup.py), the others are built from VERSIONS below.

Usage:
    ../.venv/bin/python build_notebook.py --version 04 [--out PATH]
"""
import sys
from pathlib import Path

import nbformat as nbf

VERSIONS = {
    '01': dict(images=['eff'], seeds=1, twins=False),
    '03': dict(images=['eff', 'siglip', 'clipL', 'dino'], seeds=1, twins=False),
    '04': dict(images=['eff', 'so400m', 'clipL', 'dino'], seeds=3, twins=True),
    '05': dict(images=['eff', 'so400m', 'clipL', 'dino'], seeds=3, twins=True, blend=True),
    '06': dict(images=['eff', 'so400m', 'clipL', 'dino'], seeds=3, twins=True, blend=True, dist_cuts=True),
    '07': dict(images=['eff', 'so400m', 'clipL', 'dino'], extra=['siglip', 'dinoL'], seeds=3, twins=False, blend=True,
               ensemble=True, dist_cuts=True, nbr=0.7),
    '09': dict(images=['eff', 'so400m', 'clipL', 'dino'], extra=['siglip', 'dinoL'], seeds=3, twins=False, blend=True,
               ensemble=True, dist_cuts=True, nbr=0.7, shares=[0.21, 0.25, 0.19, 0.35]),
    '10': dict(images=['eff', 'so400m', 'clipL', 'dino'], extra=['siglip', 'dinoL'],
               more=['so400m384', 'so400mall', 'giant384', 'clipL336'], seeds=3, twins=False, blend=True,
               ensemble=True, stack=True, dist_cuts=True, char_tfidf=True),
    '11': dict(images=['eff', 'so400m', 'clipL', 'dino'], extra=['siglip', 'dinoL'],
               more=['so400m384', 'so400mall', 'giant384', 'clipL336'], seeds=3, twins=False, blend=True,
               ensemble=True, stack=True, dist_cuts=True, char_tfidf=True, pp_agg=['so400m', 'clipL', 'so400m384', 'giant384']),
    '12': dict(images=['eff', 'so400m', 'clipL', 'dino'], extra=['siglip', 'dinoL'],
               more=['so400m384', 'so400mall', 'giant384', 'clipL336'], seeds=3, twins=False, blend=True,
               ensemble=True, stack=True, dist_cuts=True, char_tfidf=True, members10=True, l2_seeds=5),
    '13': dict(images=['eff', 'so400m', 'clipL', 'dino'], extra=['siglip', 'dinoL'],
               more=['so400m384', 'so400mall', 'giant384', 'clipL336'], seeds=3, twins=False, blend=True,
               ensemble=True, stack=True, dist_cuts=True, char_tfidf=True, members10=True, l2_seeds=5, species_cuts=True),
    '14': dict(images=['eff', 'so400m', 'clipL', 'dino'], extra=['siglip', 'dinoL'],
               more=['so400m384', 'so400mall', 'giant384', 'clipL336'], seeds=3, twins=False, blend=True,
               ensemble=True, stack=True, dist_cuts=True, char_tfidf=True, pp_agg=['so400m', 'clipL', 'so400m384', 'giant384'],
               no_nphotos=True),
    '15': dict(images=['eff', 'so400m', 'clipL', 'dino'], extra=['siglip', 'dinoL'],
               more=['so400m384', 'so400mall', 'giant384', 'clipL336'], seeds=3, twins=False, blend=True,
               ensemble=True, stack=True, dist_cuts=True, char_tfidf=True, pp_agg=['so400m', 'clipL', 'so400m384', 'giant384'],
               fold_test=True),
    '16': dict(images=['eff', 'so400m', 'clipL', 'dino'], extra=['siglip', 'dinoL'],
               more=['so400m384', 'so400mall', 'giant384', 'clipL336'], seeds=3, twins=False, blend=True,
               ensemble=True, stack=True, dist_cuts=True, char_tfidf=True, pp_agg=['so400m', 'clipL', 'so400m384', 'giant384'],
               n_folds=10),
    '08': dict(images=['eff', 'so400m', 'clipL', 'dino'], extra=[], more=['so400m384'], seeds=3, twins=False, blend=True,
               ensemble=True, dist_cuts=True, nbr=0.7, nbr_photo=0.95, char_tfidf=True,
               weights={'регресія': 0.31, 'класифікатор': 0.19, 'Ridge': 0.19, 'SVR': 0.31}),
}
VERSION = sys.argv[sys.argv.index('--version') + 1] if '--version' in sys.argv else '04'
CFG = VERSIONS[VERSION]
BLEND, DIST_CUTS = CFG.get('blend', False), CFG.get('dist_cuts', False)
ENSEMBLE, NBR = CFG.get('ensemble', False), CFG.get('nbr')
MORE, NBR_PHOTO, CHAR = CFG.get('more', []), CFG.get('nbr_photo'), CFG.get('char_tfidf', False)
ALLMC = ENSEMBLE and bool(CFG.get('extra'))
STACK = CFG.get('stack', False)
PP_AGG = CFG.get('pp_agg', [])
MEMBERS10, L2_SEEDS = CFG.get('members10', False), CFG.get('l2_seeds', 1)
SPECIES_CUTS = CFG.get('species_cuts', False)
NO_NPHOTOS = CFG.get('no_nphotos', False)
FOLD_TEST = CFG.get('fold_test', False)
FT = lambda line: line if FOLD_TEST else ''  # extra code only for the fold-averaged test version
DEFAULT_WEIGHTS = {'регресія': 0.28, 'класифікатор': 0.10, 'Ridge': 0.24, 'SVR': 0.24, 'класифікатор, усі фото-моделі': 0.14}
OUT = (Path(sys.argv[sys.argv.index('--out') + 1]) if '--out' in sys.argv
       else Path(__file__).parent / f'petfinder_{VERSION}.ipynb')

IMAGE_TEXT = {
    'eff': 'EfficientNet-B0, згорткова мережа, навчена на ImageNet. Без останнього шару класифікації вона повертає для фото вектор з 1280 чисел',
    'siglip': 'SigLIP 2 base (google/siglip2-base-patch16-224), трансформер, який вчився зіставляти фото з текстовими підписами. Він краще описує, що саме на фото: порода, вік, поза тварини',
    'so400m': 'SigLIP 2 so400m (google/siglip2-so400m-patch14-224), більша версія SigLIP 2 з 400 млн параметрів. Вона вчилася зіставляти фото з текстовими підписами і добре описує, що саме на фото: порода, вік, поза тварини',
    'clipL': 'CLIP ViT-L/14 (openai/clip-vit-large-patch14), ще одна модель, навчена на парах фото і тексту',
    'dino': 'DINOv2 base (facebook/dinov2-base), трансформер, який вчився лише на фото, без підписів. Беру вектор токена CLS і середній вектор патчів',
}
IMAGE_SHORT = {'eff': 'EfficientNet-B0', 'siglip': 'SigLIP 2 base', 'so400m': 'SigLIP 2 so400m', 'dinoL': 'DINOv2 large',
               'clipL': 'CLIP ViT-L/14', 'dino': 'DINOv2'}
images_list = ', '.join(IMAGE_SHORT[m] for m in CFG['images'])
cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src.strip('\n')))


def code(src):
    cells.append(nbf.v4.new_code_cell(src.strip('\n')))


plan_twins = '\n3. знайти тварин-двійників (однакове фото або однаковий опис) і перевірити, чи збігаються в них мітки;' if CFG['twins'] else ''
n = 4 if CFG['twins'] else 3
md(rf'''
# Фінальний проєкт: PetFinder.my, прогноз швидкості усиновлення

Студент: Фарід Джафарлі

Для кожної тварини є текстовий опис і фотографії. Треба передбачити AdoptionSpeed, тобто як швидко тварину заберуть з притулку. Метрика змагання: quadratic weighted kappa (QWK).

План:
1. знайти дані і подивитися на них;
2. перетворити фото і текст на числові ознаки за допомогою попередньо навчених моделей;{plan_twins}
{n}. навчити модель і перевірити її QWK на крос-валідації;
{n + 1}. перенавчити модель на всіх даних і зберегти submission.csv.
''')

md('## 1. Імпорт бібліотек')

code(rf'''
import hashlib
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb  # on macOS LightGBM has to be imported before torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import models
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, AutoTokenizer, CLIPModel, CLIPProcessor
from transformers.utils import logging as hf_logging
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if sys.platform == 'darwin':
    torch.set_num_threads(1)  # avoids an OpenMP clash between torch and LightGBM on macOS

hf_logging.disable_progress_bar()
hf_logging.set_verbosity_error()

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print('Пристрій:', DEVICE)

# settings of this version
IMAGE_MODELS = {CFG['images']!r}
N_SEEDS = {CFG['seeds']}
USE_TWINS = {CFG['twins']}
''' + (f"EXTRA_IMAGE_MODELS = {CFG['extra']!r}  # used only by one model of the blend\n" if ALLMC else '') + (f"MORE_IMAGE_MODELS = {MORE!r}  # used only by the Ridge and SVR models\n" if MORE else '') + rf'''# embeddings are saved here, so a rerun of the notebook does not compute them again
CACHE_DIR = Path(os.environ.get('EMB_CACHE', 'emb_cache'))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
''')

md(r'''
## 2. Дані

На Kaggle дані лежать у /kaggle/input, локально в папці data. Ноутбук сам знаходить таблиці train і test, а фото зіставляє з PetID за назвою файлу (PetID-1.jpg, PetID-2.jpg тощо). У таблицях лише три колонки: PetID, Description і AdoptionSpeed.
''')

code(r'''
DATA_DIR = Path('/kaggle/input') if Path('/kaggle/input').exists() else Path('data')
ID_COL, TARGET_COL, TEXT_COL = 'PetID', 'AdoptionSpeed', 'Description'


def find_csv(word):
    files = sorted(p for p in DATA_DIR.rglob('*.csv') if word in p.name.lower())
    if not files:
        raise FileNotFoundError(f'Не знайшов CSV з "{word}" у назві в {DATA_DIR}')
    return files[0]


train = pd.read_csv(find_csv('train'))
test = pd.read_csv(find_csv('test'))
for col in (ID_COL, TEXT_COL):
    assert col in train.columns and col in test.columns, f'немає колонки {col}'
train[TEXT_COL] = train[TEXT_COL].fillna('')
test[TEXT_COL] = test[TEXT_COL].fillna('')
CLASSES = sorted(int(c) for c in train[TARGET_COL].unique())
print('train:', train.shape, '| test:', test.shape, '| класи:', CLASSES)

all_ids = set(train[ID_COL]) | set(test[ID_COL])
photos = {}
for path in sorted(DATA_DIR.rglob('*')):
    if path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
        continue
    pet_id = path.stem.split('-')[0] if path.stem.split('-')[0] in all_ids else path.parent.name
    if pet_id in all_ids:
        photos.setdefault(pet_id, []).append(path)
for paths in photos.values():  # photo 1 is the main one, so sort by the number in the file name
    paths.sort(key=lambda p: int(p.stem.split('-')[-1]) if p.stem.split('-')[-1].isdigit() else 0)
for df in (train, test):
    df['n_photos'] = df[ID_COL].map(lambda p: len(photos.get(p, [])))
    print(f'Тварин з фото: {(df["n_photos"] > 0).mean():.1%}')
train.head()
''')

md('## 3. Аналіз даних')

code(r'''
counts = train[TARGET_COL].value_counts().sort_index()
print(pd.DataFrame({'тварин': counts, 'частка, %': (counts / len(train) * 100).round(1)}))

train['desc_words'] = train[TEXT_COL].str.split().str.len()
fig, axes = plt.subplots(1, 3, figsize=(17, 4))
counts.plot.bar(ax=axes[0], title='Розподіл AdoptionSpeed')
train.groupby(TARGET_COL)['desc_words'].median().plot.bar(ax=axes[1], title='Медіана слів в описі')
train.groupby(TARGET_COL)['n_photos'].mean().plot.bar(ax=axes[2], title='Середня кількість фото')
plt.tight_layout()
plt.show()
''')

code(r'''
with_photos = train[train['n_photos'] > 0].sample(6, random_state=SEED)
fig, axes = plt.subplots(1, 6, figsize=(20, 4), layout='constrained')
for ax, (_, row) in zip(axes, with_photos.iterrows()):
    ax.imshow(Image.open(photos[row[ID_COL]][0]).convert('RGB'))
    ax.set_title(f'AdoptionSpeed {row[TARGET_COL]}')
    ax.axis('off')
plt.show()
''')

if CFG['twins']:
    md(r'''
### Двійники в даних

Деякі тварини мають двійника: точно такий самий файл фото або такий самий опис (від 40 символів). Часто це одна тварина, яку виставили двічі, або тварини з одного посліду. Перевіряю, скільки таких тварин і чи збігаються в них мітки.
''')

    code(r'''
full_ids = list(train[ID_COL]) + list(test[ID_COL])
row_of = {pid: i for i, pid in enumerate(full_ids)}
by_hash, by_text = defaultdict(set), defaultdict(set)
for pid, paths in photos.items():
    for path in paths:
        by_hash[hashlib.md5(path.read_bytes()).hexdigest()].add(row_of[pid])
for i, text in enumerate(pd.concat([train[TEXT_COL], test[TEXT_COL]])):
    text = re.sub(r'\s+', ' ', text.lower()).strip()
    if len(text) >= 40:
        by_text[text].add(i)
twins = defaultdict(set)  # row -> rows of its twins (train and test together)
for group in list(by_hash.values()) + list(by_text.values()):
    for i in group:
        twins[i] |= group - {i}

n_train = len(train)
y_all = train[TARGET_COL].values
has_train_twin = np.array([any(j < n_train for j in twins[i]) for i in range(len(full_ids))])
print(f'Тварин з двійником у train: у train {has_train_twin[:n_train].sum()}, у test {has_train_twin[n_train:].sum()}')

same = [y_all[i] == round(np.mean([y_all[j] for j in twins[i] if j < n_train]))
        for i in range(n_train) if has_train_twin[i]]
print(f'У train мітка тварини збігається з міткою її двійників у {np.mean(same):.1%} випадків')
''')

md(r'''
## 4. Ознаки

### 4.1 Прості ознаки

Анкети в даних змагання немає, тому з простих ознак беру кількість фото і кількість слів в описі. Якщо в таблицях будуть інші колонки, код теж їх додасть.''' + (''' Кількість фото тут не беру: у test у тварин більше фото, ніж у train, а в train більше фото означає повільніше усиновлення. Схоже, що test зібрано пізніше, і цей зв'язок там інший.''' if NO_NPHOTOS else '') + r'''
''')

code(r'''
full = pd.concat([train.drop(columns=[TARGET_COL]), test], ignore_index=True)
if 'RescuerID' in full.columns:
    full['rescuer_count'] = full['RescuerID'].map(full['RescuerID'].value_counts())
tab_cols = [c for c in full.columns if c in test.columns and c not in (ID_COL, TEXT_COL, 'RescuerID', 'Name')]
''' + (r'''# the number of photos is left out: test pets have more photos than train pets, and in train more photos
# go with slower adoption, so this link may not hold for test
tab_cols = [c for c in tab_cols if c != 'n_photos']
''' if NO_NPHOTOS else '') + r'''tab_cols += ['rescuer_count'] if 'rescuer_count' in full.columns else []
features = full[tab_cols].copy()
for col in features.columns:
    if not pd.api.types.is_numeric_dtype(features[col]):
        features[col] = features[col].astype('category')
features['desc_words'] = full[TEXT_COL].str.split().str.len()
print('Простих ознак:', features.shape[1])


def cached(name, compute):
    # loads an embedding matrix from CACHE_DIR or computes and saves it
    path = CACHE_DIR / f'{name}.npy'
    if path.exists():
        print(f'{name}: взято з кешу')
        return np.load(path)
    emb = compute()
    np.save(path, emb)
    return emb
''')

md(r'''
### 4.2 Текст

Два представлення опису:
- TF-IDF слів і пар слів, стиснутий до 32 чисел методом TruncatedSVD;
- ембединги попередньо навченого трансформера paraphrase-multilingual-MiniLM-L12-v2. Він розуміє багато мов, а в описах трапляються і англійські, і малайські слова. Вектор опису дорівнює середньому векторів його токенів, потім PCA стискає його до 32 чисел.
''')

code(r'''
texts = full[TEXT_COL].str.lower()
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=20000, sublinear_tf=True)
svd = TruncatedSVD(32, random_state=SEED)
tfidf_matrix = tfidf.fit_transform(texts)
text_svd = svd.fit_transform(tfidf_matrix)


def minilm_embeddings():
    name = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    tokenizer = AutoTokenizer.from_pretrained(name)
    text_model = AutoModel.from_pretrained(name).to(DEVICE).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(full), 64):
            batch = tokenizer(list(full[TEXT_COL].iloc[i:i + 64]), padding=True, truncation=True,
                              max_length=256, return_tensors='pt').to(DEVICE)
            hidden = text_model(**batch).last_hidden_state
            mask = batch['attention_mask'].unsqueeze(-1)
            out.append(((hidden * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu().numpy())
    return np.vstack(out)


minilm_raw = cached('minilm', minilm_embeddings)
text_pca = PCA(32, random_state=SEED).fit_transform(minilm_raw)
for i in range(32):
    features[f'tfidf_{i}'] = text_svd[:, i]
    features[f'text_emb_{i}'] = text_pca[:, i]
print('Ознак після тексту:', features.shape[1])
''')

image_items = '\n'.join(f'- {IMAGE_TEXT[m]}{";" if k < len(CFG["images"]) - 1 else "."}'
                        for k, m in enumerate(CFG['images']))
models_word = 'одну попередньо навчену модель' if len(CFG['images']) == 1 else f'{ {2: "дві", 3: "три", 4: "чотири", 5: "п’ять"}[len(CFG["images"])]} попередньо навчені моделі'
combo = '' if len(CFG['images']) == 1 else ' Я перевіряв окремо: разом ці моделі дають кращий QWK, ніж кожна з них сама.'
md(rf'''
### 4.3 Фото

Для фото беру {models_word} без донавчання (transfer learning):
{image_items}

Для кожної тварини беру середній вектор перших трьох фото, а PCA стискає кожен вектор до 32 чисел.{combo}
''')

code(r'''
pet_ids = list(full[ID_COL])
jobs = [(row, path) for row, pid in enumerate(pet_ids) for path in photos.get(pid, [])[:3]]


def photo_embeddings(encode, batch_size=32, photo_jobs=None):
    # average vector of the first three photos of every pet (or of all photos), NaN for pets without photos
    photo_jobs = jobs if photo_jobs is None else photo_jobs
    sums, counts = None, np.zeros(len(pet_ids))
    with torch.no_grad():
        for i in range(0, len(photo_jobs), batch_size):
            rows, images = [], []
            for row, path in photo_jobs[i:i + batch_size]:
                try:
                    images.append(Image.open(path).convert('RGB'))
                    rows.append(row)
                except Exception:  # skip broken files
                    pass
            if images:
                vectors = encode(images)
                if sums is None:
                    sums = np.zeros((len(pet_ids), vectors.shape[1]), np.float32)
                for row, vec in zip(rows, vectors):
                    sums[row] += vec
                    counts[row] += 1
    emb = np.full_like(sums, np.nan)
    emb[counts > 0] = sums[counts > 0] / counts[counts > 0, None]
    return emb


def efficientnet():
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    cnn = models.efficientnet_b0(weights=weights)
    cnn.classifier = torch.nn.Identity()
    cnn = cnn.to(DEVICE).eval()
    preprocess = weights.transforms()
    return photo_embeddings(lambda ims: cnn(torch.stack([preprocess(i) for i in ims]).to(DEVICE)).cpu().numpy())


def text_image_model(name, model_cls=AutoModel, processor_cls=AutoProcessor, photo_jobs=None):
    # SigLIP and CLIP: normalized image vector from get_image_features
    processor = processor_cls.from_pretrained(name)
    model = model_cls.from_pretrained(name).to(DEVICE).eval()

    def encode(images):
        out = model.get_image_features(**processor(images=images, return_tensors='pt').to(DEVICE))
        out = out.pooler_output if hasattr(out, 'pooler_output') else out
        return torch.nn.functional.normalize(out, dim=-1).float().cpu().numpy()
    return photo_embeddings(encode, photo_jobs=photo_jobs)


def dinov2(name='facebook/dinov2-base'):
    processor = AutoImageProcessor.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(DEVICE).eval()

    def encode(images):
        hidden = model(**processor(images=images, return_tensors='pt').to(DEVICE)).last_hidden_state
        return torch.cat([hidden[:, 0], hidden[:, 1:].mean(1)], 1).cpu().numpy()
    return photo_embeddings(encode)


IMAGE_ENCODERS = {
    'eff': efficientnet,
    'siglip': lambda: text_image_model('google/siglip2-base-patch16-224'),
    'so400m': lambda: text_image_model('google/siglip2-so400m-patch14-224'),
    'clipL': lambda: text_image_model('openai/clip-vit-large-patch14', CLIPModel, CLIPProcessor),
    'dino': dinov2,
    'dinoL': lambda: dinov2('facebook/dinov2-large'),
    'so400m384': lambda: text_image_model('google/siglip2-so400m-patch14-384'),
    'giant384': lambda: text_image_model('google/siglip2-giant-opt-patch16-384'),
    'clipL336': lambda: text_image_model('openai/clip-vit-large-patch14-336', CLIPModel, CLIPProcessor),
    # so400m averaged over all photos of the pet, not only the first three
    'so400mall': lambda: text_image_model('google/siglip2-so400m-patch14-224', photo_jobs=[
        (row, path) for row, pid in enumerate(pet_ids) for path in photos.get(pid, [])]),
}


def add_pca(features, emb, prefix):
    has = ~np.isnan(emb[:, 0])
    reduced = np.full((len(emb), 32), np.nan)
    reduced[has] = PCA(32, random_state=SEED).fit_transform(emb[has])
    return pd.concat([features, pd.DataFrame(reduced, columns=[f'{prefix}_{i}' for i in range(32)])], axis=1)


raw_emb = {}  # full vectors, the Ridge models use them
for name in IMAGE_MODELS:
    emb = cached(name, IMAGE_ENCODERS[name])
    raw_emb[name] = emb
    features = add_pca(features, emb, f'img_{name}')
print(f'Фото: {len(jobs)}, тварин з ембедингами фото: {(~np.isnan(emb[:, 0])).mean():.1%}')
print('Усього ознак:', features.shape[1])
''' + (r'''
features_all = features.copy()  # the same features plus two more photo models
for name in EXTRA_IMAGE_MODELS:
    features_all = add_pca(features_all, cached(name, IMAGE_ENCODERS[name]), f'img_{name}')
print('Ознак для класифікатора з усіма фото-моделями:', features_all.shape[1])
''' if ALLMC else '') + (r'''
raw_emb_more = {name: cached(name, IMAGE_ENCODERS[name]) for name in MORE_IMAGE_MODELS}
''' if MORE else (r'''
raw_emb_more = {}
''' if ENSEMBLE else '')))

seeds_text = (f' Модель навчаю з {CFG["seeds"]} різними seed і усереднюю прогнози, так вони стабільніші.'
              if CFG['seeds'] > 1 else '')
twins_text = (r'''

Для тварин, у яких є двійник у навчальній частині, змішую прогноз моделі навпіл із середньою міткою двійників. На крос-валідації двійників шукаю лише в навчальних фолдах, щоб оцінка була чесною.''' if CFG['twins'] else '')
blend_text = (r'''

Крім LightGBM-регресії навчаю ще дві моделі:
- LightGBM-класифікатор на 4 класи. З його ймовірностей рахую очікуваний клас, тобто теж число від 1 до 4;
- лінійну регресію Ridge окремо на кожному повному векторі (TF-IDF, MiniLM і кожна модель для фото), а їхні прогнози усереднюю.

Підсумковий прогноз дорівнює 0.5 × регресія + 0.3 × класифікатор + 0.2 × Ridge. Ваги я підібрав на out-of-fold прогнозах. Моделі помиляються по-різному, тому суміш точніша за кожну з них.''' if BLEND and not ENSEMBLE else '')
if ENSEMBLE:
    blend_text = r'''

Крім LightGBM-регресії навчаю ще чотири моделі:
- LightGBM-класифікатор на 4 класи. З його ймовірностей рахую очікуваний клас, тобто теж число від 1 до 4;
- лінійну регресію Ridge окремо на кожному повному векторі (TF-IDF, MiniLM і кожна модель для фото), а їхні прогнози усереднюю;
- регресію методом опорних векторів (SVR з RBF-ядром) окремо на кожному повному векторі фото, прогнози теж усереднюю;
- ще один LightGBM-класифікатор, який бачить на дві моделі для фото більше (SigLIP 2 base і DINOv2 large).

Ваги суміші я підібрав на out-of-fold прогнозах. Моделі помиляються по-різному, тому суміш точніша за кожну з них.'''
    if not ALLMC:
        blend_text = r'''

Крім LightGBM-регресії навчаю ще три моделі:
- LightGBM-класифікатор на 4 класи. З його ймовірностей рахую очікуваний клас, тобто теж число від 1 до 4;
- лінійну регресію Ridge окремо на кожному повному векторі (символьні n-грами опису, MiniLM і кожна модель для фото), а їхні прогнози усереднюю;
- регресію методом опорних векторів (SVR з RBF-ядром) окремо на кожному повному векторі фото, прогнози теж усереднюю.

Для Ridge і SVR додаю ще SigLIP 2 so400m з більшою роздільністю 384 пікселі. Ваги суміші я підібрав на out-of-fold прогнозах. Моделі помиляються по-різному, тому суміш точніша за кожну з них.'''
if STACK:
    blend_text = r'''

Крім LightGBM-регресії навчаю ще чотири моделі:
- LightGBM-класифікатор на 4 класи. З його ймовірностей рахую очікуваний клас, тобто теж число від 1 до 4;
- лінійну регресію Ridge окремо на кожному повному векторі (символьні n-грами опису, MiniLM і кожна модель для фото), а їхні прогнози усереднюю;
- регресію методом опорних векторів (SVR з RBF-ядром) окремо на кожному повному векторі фото, прогнози теж усереднюю;
- ще один LightGBM-класифікатор, який бачить на дві моделі для фото більше (SigLIP 2 base і DINOv2 large).

Для Ridge і SVR беру ще чотири варіанти векторів фото: SigLIP 2 so400m з роздільністю 384, SigLIP 2 so400m, усереднений за всіма фото тварини, SigLIP 2 giant і CLIP ViT-L/14 з роздільністю 336.

Ці п'ять прогнозів не змішую вручну. Їх отримує невелика LightGBM другого рівня (stacking) разом з ознаками сусідів: середня мітка тварин з train з дуже схожим описом або фото, найбільша схожість, кількість таких сусідів і мітка найближчого. Багато оголошень написані за одним шаблоном, часто тим самим волонтером, і в таких тварин мітки часто схожі. Друга модель сама вирішує, коли варто довіряти сусідам. На крос-валідації сусідів шукаю лише в навчальних фолдах.'''
    if MEMBERS10:
        blend_text = blend_text.replace('Ці п\'ять прогнозів не змішую вручну.', '''Ще п'ять моделей додаю лише для різноманітності: LightGBM з випадковими точками розбиття (extra trees), CatBoost, k найближчих сусідів за векторами фото, логістичну регресію на 4 класи і Ridge, навчений на кожному фото окремо (оцінка тварини дорівнює середній оцінці її фото).

Ці десять прогнозів не змішую вручну.''').replace('Друга модель сама вирішує, коли варто довіряти сусідам.', 'Друга модель сама вирішує, коли варто довіряти сусідам. Її навчаю з п\'ятьма різними seed і усереднюю.')
    if PP_AGG:
        blend_text += r'''

Ще друга модель бачить, як оцінюється кожне фото окремо. Для чотирьох моделей фото навчаю Ridge на окремих фото (кожне фото отримує мітку своєї тварини), а для тварини беру середню, найбільшу і найменшу оцінку її фото, розкид оцінок і оцінку першого фото.'''
nbr_text = (rf'''

Схожі описи. Багато оголошень написані за одним шаблоном, часто тим самим волонтером. Якщо опис тварини дуже схожий на опис тварини з train (схожість TF-IDF за символами від {NBR}), змішую прогноз навпіл із середньою міткою таких сусідів, зваженою за схожістю. На крос-валідації сусідів шукаю лише в навчальних фолдах.''' if NBR else '')
if NBR_PHOTO:
    nbr_text += rf''' Так само сусідом вважаю тварину з train, у якої хоча б одне фото майже таке саме (схожість векторів SigLIP 2 so400m від {NBR_PHOTO}). Для цього рахую вектори всіх фото, а не лише перших трьох.'''
md(rf'''
## 5. Модель і крос-валідація

Модель: LightGBM-регресія. Вона передбачає число (наприклад, 2.3), а пороги перетворюють його на клас від найменшого до найбільшого. Пороги підбираю так, щоб QWK був максимальним.{seeds_text}

Щоб модель не перенавчалася, використовую регуляризацію L2 (lambda_l2), мінімальну кількість тварин у листі, випадкову підвибірку ознак і рядків та ранню зупинку за валідаційною частиною кожного фолду.

Якість перевіряю {CFG.get('n_folds', 5)}-кратною стратифікованою крос-валідацією: кожна тварина один раз потрапляє у валідацію, тож прогнози є для всього train (out-of-fold).{blend_text}{twins_text}{nbr_text}
''')

code(r'''
X = features.iloc[:len(train)].reset_index(drop=True)
X_test = features.iloc[len(train):].reset_index(drop=True)
y = train[TARGET_COL].values

PARAMS = dict(objective='regression', learning_rate=0.02, num_leaves=31, min_child_samples=30,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              deterministic=True, force_row_wise=True, verbose=-1, n_jobs=4)


def qwk(a, b):
    return cohen_kappa_score(a, b, weights='quadratic')


def to_classes(pred, cuts):
    return np.array(CLASSES)[np.digitize(pred, cuts)]


def fit_cuts(pred, target):
    cuts = np.array(CLASSES[:-1]) + 0.5  # start from the middle points between classes
    for _ in range(3):
        for k in range(len(cuts)):
            grid = np.linspace(cuts[k] - 0.5, cuts[k] + 0.5, 41)
            scores = [qwk(target, to_classes(pred, np.sort(np.r_[cuts[:k], g, cuts[k + 1:]]))) for g in grid]
            cuts[k] = grid[int(np.argmax(scores))]
            cuts = np.sort(cuts)
    return cuts


oof = np.zeros(len(X))
best_iters = []
folds = list(StratifiedKFold(n_splits=''' + str(CFG.get('n_folds', 5)) + r''', shuffle=True, random_state=SEED).split(X, y))
''' + FT('fold_test_lgb = np.zeros(len(X_test))  # test predictions of the fold models, used by the second level\n') + r'''
for fold, (tr, va) in enumerate(folds, 1):
    for s in range(N_SEEDS):
        model = lgb.train({**PARAMS, 'seed': SEED + s}, lgb.Dataset(X.iloc[tr], y[tr]), num_boost_round=5000,
                          valid_sets=[lgb.Dataset(X.iloc[va], y[va])],
                          callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[va] += model.predict(X.iloc[va], num_iteration=model.best_iteration) / N_SEEDS
''' + FT('        fold_test_lgb += model.predict(X_test, num_iteration=model.best_iteration) / (N_SEEDS * len(folds))\n') + r'''        best_iters.append(model.best_iteration)
    print(f'Фолд {fold}: дерев {np.mean(best_iters[-N_SEEDS:]):.0f}, QWK з базовими порогами '
          f'{qwk(y[va], to_classes(oof[va], np.array(CLASSES[:-1]) + 0.5)):.4f}')
''')

if BLEND:
    code(r'''
from sklearn.linear_model import Ridge

MC_PARAMS = {**PARAMS, 'objective': 'multiclass', 'num_class': len(CLASSES)}
y_idx = np.searchsorted(CLASSES, y)
''' + (r'''ridge_inputs = {'tfidf': (tfidf_matrix, 3.0)}
for name, emb in [('minilm', minilm_raw)] + list(raw_emb.items()):''' if not CHAR else r'''# character n-grams catch the writing style of the description better than words
char_ngrams = TfidfVectorizer(analyzer='char', ngram_range=(2, 6), min_df=3, max_features=300000, sublinear_tf=True)
ridge_inputs = {'tfidf': (char_ngrams.fit_transform(texts), 1.0)}
for name, emb in [('minilm', minilm_raw)] + list(raw_emb.items()) + list(raw_emb_more.items()):''') + r'''
    z = np.nan_to_num(emb)
    z = (z - z.mean(0)) / (z.std(0) + 1e-6)
    ridge_inputs[name] = (z, 300.0 * z.shape[1] / 768)  # longer vectors get stronger regularization

oof_lgb, oof_mc, oof_ridge = oof.copy(), np.zeros(len(X)), np.zeros(len(X))
''' + FT('fold_test_mc, fold_test_ridge = np.zeros(len(X_test)), np.zeros(len(X_test))\n') + r'''mc_iters = []
for tr, va in folds:
    for s in range(N_SEEDS):
        mc = lgb.train({**MC_PARAMS, 'seed': SEED + s}, lgb.Dataset(X.iloc[tr], y_idx[tr]), num_boost_round=5000,
                       valid_sets=[lgb.Dataset(X.iloc[va], y_idx[va])],
                       callbacks=[lgb.early_stopping(200, verbose=False)])
        oof_mc[va] += mc.predict(X.iloc[va], num_iteration=mc.best_iteration) @ np.array(CLASSES) / N_SEEDS
''' + FT('        fold_test_mc += mc.predict(X_test, num_iteration=mc.best_iteration) @ np.array(CLASSES) / (N_SEEDS * len(folds))\n') + r'''        mc_iters.append(mc.best_iteration)
    for z, alpha in ridge_inputs.values():
        z_train = z[:len(X)]
''' + (r'''        oof_ridge[va] += Ridge(alpha=alpha).fit(z_train[tr], y[tr]).predict(z_train[va]) / len(ridge_inputs)
''' if not FOLD_TEST else r'''        r = Ridge(alpha=alpha).fit(z_train[tr], y[tr])
        oof_ridge[va] += r.predict(z_train[va]) / len(ridge_inputs)
        fold_test_ridge += r.predict(z[len(X):]) / (len(ridge_inputs) * len(folds))
''') + r'''
''' + (r'''WEIGHTS = (0.5, 0.3, 0.2)
for name, p in (('регресія', oof_lgb), ('класифікатор', oof_mc), ('Ridge', oof_ridge)):
    print(f'OOF QWK, {name}: {qwk(y, to_classes(p, fit_cuts(p, y))):.4f}')
oof = WEIGHTS[0] * oof_lgb + WEIGHTS[1] * oof_mc + WEIGHTS[2] * oof_ridge
print(f'OOF QWK суміші трьох моделей: {qwk(y, to_classes(oof, fit_cuts(oof, y))):.4f}')
''' if not ENSEMBLE else r'''from sklearn.svm import SVR

svr_inputs = []
for emb in list(raw_emb.values()) + list(raw_emb_more.values()):
    z = np.nan_to_num(emb)
    z = PCA(256, random_state=SEED).fit_transform((z - z.mean(0)) / (z.std(0) + 1e-6))
    svr_inputs.append(z / np.sqrt(z.shape[1]))
''' + (r'''X_all = features_all.iloc[:len(train)].reset_index(drop=True)
X_all_test = features_all.iloc[len(train):].reset_index(drop=True)

oof_svr, oof_allmc = np.zeros(len(X)), np.zeros(len(X))
allmc_iters = []
''' + FT('fold_test_svr, fold_test_allmc = np.zeros(len(X_test)), np.zeros(len(X_test))\n') + r'''for tr, va in folds:
    for z in svr_inputs:
''' + (r'''        oof_svr[va] += SVR(C=1.0, epsilon=0.3).fit(z[:len(X)][tr], y[tr]).predict(z[:len(X)][va]) / len(svr_inputs)''' if not FOLD_TEST else r'''        r = SVR(C=1.0, epsilon=0.3).fit(z[:len(X)][tr], y[tr])
        oof_svr[va] += r.predict(z[:len(X)][va]) / len(svr_inputs)
        fold_test_svr += r.predict(z[len(X):]) / (len(svr_inputs) * len(folds))''') + r'''
    for s in range(N_SEEDS):''' if ALLMC else r'''oof_svr = np.zeros(len(X))
for tr, va in folds:
    for z in svr_inputs:
        oof_svr[va] += SVR(C=1.0, epsilon=0.3).fit(z[:len(X)][tr], y[tr]).predict(z[:len(X)][va]) / len(svr_inputs)
''') + (r'''
        m = lgb.train({**MC_PARAMS, 'seed': SEED + s}, lgb.Dataset(X_all.iloc[tr], y_idx[tr]), num_boost_round=5000,
                      valid_sets=[lgb.Dataset(X_all.iloc[va], y_idx[va])],
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        oof_allmc[va] += m.predict(X_all.iloc[va], num_iteration=m.best_iteration) @ np.array(CLASSES) / N_SEEDS
''' + FT('        fold_test_allmc += m.predict(X_all_test, num_iteration=m.best_iteration) @ np.array(CLASSES) / (N_SEEDS * len(folds))\n') + r'''        allmc_iters.append(m.best_iteration)
''' if ALLMC else '') + r'''
oof_models = {'регресія': oof_lgb, 'класифікатор': oof_mc, 'Ridge': oof_ridge, 'SVR': oof_svr''' + (r''',
              'класифікатор, усі фото-моделі': oof_allmc}''' if ALLMC else '}') + ('\nWEIGHTS = ' + repr(CFG.get('weights', DEFAULT_WEIGHTS)) if not STACK else '') + r'''
for name, p in oof_models.items():
    print(f'OOF QWK, {name}: {qwk(y, to_classes(p, fit_cuts(p, y))):.4f}')
''' + (r'''oof = sum(WEIGHTS[k] * oof_models[k] for k in WEIGHTS)
print(f'OOF QWK суміші: {qwk(y, to_classes(oof, fit_cuts(oof, y))):.4f}')''' if not STACK else '') + r'''
''') + r'''
''')

if MEMBERS10:
    code("\n# five more models for the second level: different kinds of models make different mistakes\nfrom catboost import CatBoostRegressor\nfrom sklearn.linear_model import LogisticRegression\n\nET_PARAMS = {**PARAMS, 'extra_trees': True, 'min_child_samples': 20}  # LightGBM with random split points\nCAT_PARAMS = dict(loss_function='RMSE', depth=6, learning_rate=0.03, l2_leaf_reg=3, random_seed=SEED, verbose=0, thread_count=4)\noof_et, oof_cat, et_iters, cat_iters = np.zeros(len(X)), np.zeros(len(X)), [], []\nfor tr, va in folds:\n    for s in range(N_SEEDS):\n        m = lgb.train({**ET_PARAMS, 'seed': SEED + s}, lgb.Dataset(X.iloc[tr], y[tr]), num_boost_round=5000,\n                      valid_sets=[lgb.Dataset(X.iloc[va], y[va])], callbacks=[lgb.early_stopping(200, verbose=False)])\n        oof_et[va] += m.predict(X.iloc[va], num_iteration=m.best_iteration) / N_SEEDS\n        et_iters.append(m.best_iteration)\n    cb = CatBoostRegressor(iterations=5000, od_type='Iter', od_wait=200, **CAT_PARAMS)\n    cb.fit(X.iloc[tr], y[tr], eval_set=(X.iloc[va], y[va]), use_best_model=True)\n    oof_cat[va] = cb.predict(X.iloc[va])\n    cat_iters.append(cb.get_best_iteration())\ntest_et = np.zeros(len(X_test))\nfor s in range(N_SEEDS):\n    m = lgb.train({**ET_PARAMS, 'seed': SEED + s}, lgb.Dataset(X, y), num_boost_round=int(np.mean(et_iters) * 1.1))\n    test_et += m.predict(X_test) / N_SEEDS\ntest_cat = CatBoostRegressor(iterations=int(np.mean(cat_iters) * 1.1), **CAT_PARAMS).fit(X, y).predict(X_test)\n\n# k nearest train pets by photo vectors (so400m + CLIP), closer pets count more\nboth_raw = np.hstack([np.nan_to_num(raw_emb['so400m']), np.nan_to_num(raw_emb['clipL'])])\nboth_raw /= np.linalg.norm(both_raw, axis=1, keepdims=True) + 1e-9\n\n\ndef knn_predict(pool, rows, k=50):\n    s = both_raw[rows] @ both_raw[pool].T\n    idx = np.argpartition(-s, k, axis=1)[:, :k]\n    w = np.exp(np.take_along_axis(s, idx, 1) * 10)\n    return (y[pool][idx] * w).sum(1) / w.sum(1)\n\n\noof_knn = np.zeros(len(X))\nfor tr, va in folds:\n    oof_knn[va] = knn_predict(tr, va)\ntest_knn = knn_predict(np.arange(len(train)), np.arange(len(train), len(full)))\n\n# logistic regression on 4 classes over all photo vectors together, expected class\nlr_input = []\nfor emb in raw_emb.values():\n    z = np.nan_to_num(emb)\n    lr_input.append(PCA(128, random_state=SEED).fit_transform((z - z.mean(0)) / (z.std(0) + 1e-6)))\nlr_input = np.hstack(lr_input)\nlr_input = (lr_input - lr_input.mean(0)) / (lr_input.std(0) + 1e-6)\noof_logreg = np.zeros(len(X))\nfor tr, va in folds:\n    oof_logreg[va] = LogisticRegression(C=0.01, max_iter=2000).fit(lr_input[tr], y[tr]).predict_proba(lr_input[va]) @ np.array(CLASSES)\ntest_logreg = (LogisticRegression(C=0.01, max_iter=2000).fit(lr_input[:len(X)], y)\n               .predict_proba(lr_input[len(X):]) @ np.array(CLASSES))\n\n# Ridge on every photo separately (each photo gets the label of its pet), pet score = mean over its photos\nPP_MODELS = {'so400m': ('google/siglip2-so400m-patch14-224', AutoModel, AutoProcessor),\n             'clipL': ('openai/clip-vit-large-patch14', CLIPModel, CLIPProcessor),\n             'so400m384': ('google/siglip2-so400m-patch14-384', AutoModel, AutoProcessor),\n             'giant384': ('google/siglip2-giant-opt-patch16-384', AutoModel, AutoProcessor)}\nall_jobs = [(row, path) for row, pid in enumerate(pet_ids) for path in photos.get(pid, [])]\nphoto_pet = np.array([row for row, _ in all_jobs])\n\n\ndef photo_vectors(name):\n    path = CACHE_DIR / f'{name}_allphotos.npy'\n    if path.exists():\n        print(f'{name}_allphotos: взято з кешу')\n        return np.load(path).astype(np.float32)\n    model_name, model_cls, processor_cls = PP_MODELS[name]\n    processor = processor_cls.from_pretrained(model_name)\n    model = model_cls.from_pretrained(model_name).to(DEVICE).eval()\n    vecs = []\n    with torch.no_grad():\n        for i in range(0, len(all_jobs), 32):\n            images = [Image.open(p).convert('RGB') for _, p in all_jobs[i:i + 32]]\n            out = model.get_image_features(**processor(images=images, return_tensors='pt').to(DEVICE))\n            out = out.pooler_output if hasattr(out, 'pooler_output') else out\n            vecs.append(torch.nn.functional.normalize(out, dim=-1).float().cpu().numpy())\n    vecs = np.vstack(vecs)\n    np.save(path, vecs.astype(np.float16))\n    return vecs\n\n\npet_means = []\nfor name in PP_MODELS:\n    z = photo_vectors(name)\n    z = (z - z.mean(0)) / (z.std(0) + 1e-6)\n    pred = np.full(len(photo_pet), np.nan)\n    for tr, va in folds:\n        in_tr, in_va = np.isin(photo_pet, tr), np.isin(photo_pet, va)\n        pred[in_va] = Ridge(alpha=10000).fit(z[in_tr], y[photo_pet[in_tr]]).predict(z[in_va])\n    in_tr, in_te = photo_pet < len(train), photo_pet >= len(train)\n    pred[in_te] = Ridge(alpha=10000).fit(z[in_tr], y[photo_pet[in_tr]]).predict(z[in_te])\n    pet_means.append(pd.Series(pred).groupby(photo_pet).mean().reindex(range(len(full))).values)\npp = np.nanmean(pet_means, axis=0)\noof_pp, test_pp = pp[:len(train)].copy(), pp[len(train):].copy()\noof_pp[np.isnan(oof_pp)] = np.nanmean(oof_pp)  # 4 pets without photos\ntest_pp[np.isnan(test_pp)] = np.nanmean(test_pp)\n\nmore_oof = {'LightGBM extra trees': oof_et, 'CatBoost': oof_cat, 'kNN': oof_knn, 'логістична регресія': oof_logreg,\n            'Ridge по окремих фото': oof_pp}\nmore_test = {'LightGBM extra trees': test_et, 'CatBoost': test_cat, 'kNN': test_knn, 'логістична регресія': test_logreg,\n             'Ridge по окремих фото': test_pp}\nfor name, p in more_oof.items():\n    print(f'OOF QWK, {name}: {qwk(y, to_classes(p, fit_cuts(p, y))):.4f}')\noof_models.update(more_oof)\n")

if STACK:
    code(r'''
# neighbours: very similar descriptions (character TF-IDF) and very similar photos (so400m + CLIP vectors)
chars = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2, sublinear_tf=True).fit_transform(full[TEXT_COL].str.lower())
text_sim = (chars @ chars[:len(train)].T).toarray().astype(np.float32)
text_sim[full[TEXT_COL].str.len().values < 40] = 0
both = np.hstack([np.nan_to_num(raw_emb[n]) / (np.linalg.norm(np.nan_to_num(raw_emb[n]), axis=1, keepdims=True) + 1e-9)
                  for n in ('so400m', 'clipL')]) / np.sqrt(2)
photo_sim = (both @ both[:len(train)].T).astype(np.float32)
for m in (text_sim, photo_sim):
    m[np.arange(len(train)), np.arange(len(train))] = 0


def neighbour_features(rows, pool):
    # for every row: weighted mean label of similar train pets in pool, best similarity, count, label of the closest one
    out = {}
    for name, sim_all, thr in (('text', text_sim, 0.5), ('photo', photo_sim, 0.9)):
        s = sim_all[np.ix_(rows, pool)]
        w = np.where(s >= thr, s, 0)
        total = w.sum(1)
        out[f'{name}_label'] = np.where(total > 0, (w @ y[pool]) / np.maximum(total, 1e-9), np.nan)
        out[f'{name}_max'] = s.max(1)
        out[f'{name}_count'] = (s >= thr).sum(1)
        out[f'{name}_top_label'] = y[pool][s.argmax(1)]
    return pd.DataFrame(out)


# second level: a small LightGBM learns how to combine the five models and the neighbours
parts = []
for tr, va in folds:
    part = neighbour_features(va, tr)
    part.index = va
    parts.append(part)
# LightGBM needs plain feature names
L2_NAMES = {'регресія': 'lgb_regression', 'класифікатор': 'lgb_classifier', 'Ridge': 'ridge', 'SVR': 'svr',
            'класифікатор, усі фото-моделі': 'lgb_classifier_all_photo_models', 'LightGBM extra trees': 'lgb_extra_trees',
            'CatBoost': 'catboost', 'kNN': 'knn', 'логістична регресія': 'logreg', 'Ridge по окремих фото': 'ridge_per_photo'}
level2 = pd.concat([pd.DataFrame(oof_models).rename(columns=L2_NAMES), pd.concat(parts).sort_index()], axis=1)
''' + ("level2['n_photos'] = train['n_photos'].values" if not NO_NPHOTOS else '') + (r'''

# every photo separately: a Ridge model per photo model, then statistics of the photo predictions of each pet
PP_MODELS = {'so400m': ('google/siglip2-so400m-patch14-224', AutoModel, AutoProcessor),
             'clipL': ('openai/clip-vit-large-patch14', CLIPModel, CLIPProcessor),
             'so400m384': ('google/siglip2-so400m-patch14-384', AutoModel, AutoProcessor),
             'giant384': ('google/siglip2-giant-opt-patch16-384', AutoModel, AutoProcessor)}
all_jobs = [(row, path) for row, pid in enumerate(pet_ids) for path in photos.get(pid, [])]
photo_pet = np.array([row for row, _ in all_jobs])
is_first = np.r_[True, photo_pet[1:] != photo_pet[:-1]]


def photo_vectors(name):
    path = CACHE_DIR / f'{name}_allphotos.npy'
    if path.exists():
        print(f'{name}_allphotos: взято з кешу')
        return np.load(path).astype(np.float32)
    model_name, model_cls, processor_cls = PP_MODELS[name]
    processor = processor_cls.from_pretrained(model_name)
    model = model_cls.from_pretrained(model_name).to(DEVICE).eval()
    vecs = []
    with torch.no_grad():
        for i in range(0, len(all_jobs), 32):
            images = [Image.open(p).convert('RGB') for _, p in all_jobs[i:i + 32]]
            out = model.get_image_features(**processor(images=images, return_tensors='pt').to(DEVICE))
            out = out.pooler_output if hasattr(out, 'pooler_output') else out
            vecs.append(torch.nn.functional.normalize(out, dim=-1).float().cpu().numpy())
    vecs = np.vstack(vecs)
    np.save(path, vecs.astype(np.float16))
    return vecs


photo_stats = {}
for name in PP_MODELS:
    z = photo_vectors(name)
    z = (z - z.mean(0)) / (z.std(0) + 1e-6)
    pred = np.full(len(photo_pet), np.nan)
    for tr, va in folds:  # each photo is predicted by a model that did not see its pet
        in_tr, in_va = np.isin(photo_pet, tr), np.isin(photo_pet, va)
''' + (r'''        pred[in_va] = Ridge(alpha=10000).fit(z[in_tr], y[photo_pet[in_tr]]).predict(z[in_va])
    in_tr, in_te = photo_pet < len(train), photo_pet >= len(train)
    pred[in_te] = Ridge(alpha=10000).fit(z[in_tr], y[photo_pet[in_tr]]).predict(z[in_te])
''' if not FOLD_TEST else r'''        r = Ridge(alpha=10000).fit(z[in_tr], y[photo_pet[in_tr]])
        pred[in_va] = r.predict(z[in_va])
        in_te = photo_pet >= len(train)
        pred[in_te] = np.nan_to_num(pred[in_te]) + r.predict(z[in_te]) / len(folds)  # mean of the fold models on test photos
''') + r'''    g = pd.DataFrame({'pet': photo_pet, 'p': pred}).groupby('pet').p
    stats = pd.DataFrame({f'{name}_mean': g.mean(), f'{name}_max': g.max(), f'{name}_min': g.min(), f'{name}_std': g.std()})
    stats[f'{name}_first'] = pd.Series(pred[is_first], index=photo_pet[is_first])
    for col in stats:
        photo_stats[col] = stats[col].reindex(range(len(full))).values
photo_stats = pd.DataFrame(photo_stats)
level2 = pd.concat([level2, photo_stats.iloc[:len(train)].reset_index(drop=True)], axis=1)
print('Ознак другого рівня:', level2.shape[1])
''' if PP_AGG else '') + r'''
L2_PARAMS = dict(objective='regression', learning_rate=0.01, num_leaves=7, min_child_samples=50, feature_fraction=0.8,
                 bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, n_jobs=4, deterministic=True, force_row_wise=True)
oof = np.zeros(len(X))
l2_iters = []
''' + (r'''for tr, va in folds:
    m = lgb.train(L2_PARAMS, lgb.Dataset(level2.iloc[tr], y[tr]), num_boost_round=3000,
                  valid_sets=[lgb.Dataset(level2.iloc[va], y[va])], callbacks=[lgb.early_stopping(200, verbose=False)])
    oof[va] = m.predict(level2.iloc[va], num_iteration=m.best_iteration)
    l2_iters.append(m.best_iteration)''' if L2_SEEDS == 1 else f'''L2_SEEDS = {L2_SEEDS}  # several seeds of the second level, their mean is more stable''' + r'''
for tr, va in folds:
    for s in range(L2_SEEDS):
        m = lgb.train({**L2_PARAMS, 'seed': s}, lgb.Dataset(level2.iloc[tr], y[tr]), num_boost_round=3000,
                      valid_sets=[lgb.Dataset(level2.iloc[va], y[va])], callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[va] += m.predict(level2.iloc[va], num_iteration=m.best_iteration) / L2_SEEDS
        l2_iters.append(m.best_iteration)''') + r'''
print(f'OOF QWK другого рівня: {qwk(y, to_classes(oof, fit_cuts(oof, y))):.4f}')
''')

if NBR:
    code(rf'''
NBR_SIM, NBR_WEIGHT = {NBR}, 0.5
char_tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2, sublinear_tf=True)
chars = char_tfidf.fit_transform(full[TEXT_COL].str.lower())
sim = (chars @ chars[:len(train)].T).toarray().astype(np.float32)  # every pet against every train pet
sim[full[TEXT_COL].str.len().values < 40] = 0  # very short descriptions are too generic
sim[np.arange(len(train)), np.arange(len(train))] = 0
''' + (rf'''

# photos: one so400m vector for every photo of every pet, then for each pair of pets the most similar pair of photos
NBR_PHOTO_SIM = {NBR_PHOTO}


def all_photo_embeddings():
    processor = AutoProcessor.from_pretrained('google/siglip2-so400m-patch14-224')
    model = AutoModel.from_pretrained('google/siglip2-so400m-patch14-224').to(DEVICE).eval()
    all_jobs = [(row, path) for row, pid in enumerate(pet_ids) for path in photos.get(pid, [])]
    vecs = []
    with torch.no_grad():
        for i in range(0, len(all_jobs), 32):
            images = [Image.open(path).convert('RGB') for _, path in all_jobs[i:i + 32]]
            out = model.get_image_features(**processor(images=images, return_tensors='pt').to(DEVICE))
            out = out.pooler_output if hasattr(out, 'pooler_output') else out
            vecs.append(torch.nn.functional.normalize(out, dim=-1).float().cpu().numpy())
    return np.vstack(vecs), np.array([row for row, _ in all_jobs])


if (CACHE_DIR / 'so400m_allphotos.npy').exists():
    print('so400m_allphotos: взято з кешу')
    photo_vecs, photo_rows = np.load(CACHE_DIR / 'so400m_allphotos.npy'), np.load(CACHE_DIR / 'so400m_allphotos_pet.npy')
else:
    photo_vecs, photo_rows = all_photo_embeddings()
    np.save(CACHE_DIR / 'so400m_allphotos.npy', photo_vecs.astype(np.float16))
    np.save(CACHE_DIR / 'so400m_allphotos_pet.npy', photo_rows)
photo_vecs = photo_vecs.astype(np.float32)
photo_vecs /= np.linalg.norm(photo_vecs, axis=1, keepdims=True)
order = np.argsort(photo_rows, kind='stable')
photo_vecs, photo_rows = photo_vecs[order], photo_rows[order]
tr_vecs, tr_rows = photo_vecs[photo_rows < len(train)], photo_rows[photo_rows < len(train)]
tr_starts = np.r_[0, np.flatnonzero(np.diff(tr_rows)) + 1]
photo_sim = np.zeros((len(full), len(train)), np.float32)
for pets in np.array_split(np.unique(photo_rows), 20):
    in_chunk = np.isin(photo_rows, pets)
    q_rows = photo_rows[in_chunk]
    best = np.maximum.reduceat(photo_vecs[in_chunk] @ tr_vecs.T, tr_starts, axis=1)  # photo x train pet
    starts = np.r_[0, np.flatnonzero(np.diff(q_rows)) + 1]
    photo_sim[np.ix_(q_rows[starts], tr_rows[tr_starts])] = np.maximum.reduceat(best, starts, axis=0)
photo_sim[np.arange(len(train)), np.arange(len(train))] = 0
print('Фото всіх тварин:', len(photo_vecs))
''' if NBR_PHOTO else '') + rf'''


def neighbour_label(rows, pool):
    # similarity-weighted mean label of train pets in pool whose description''' + (''' or one of the photos''' if NBR_PHOTO else '') + rf''' is very similar
    w = sim[np.ix_(rows, pool)]
    w = np.where(w >= NBR_SIM, w, 0)
''' + (r'''    p = photo_sim[np.ix_(rows, pool)]
    w = w + np.where(p >= NBR_PHOTO_SIM, p, 0)
''' if NBR_PHOTO else '') + rf'''    total = w.sum(1)
    out = np.full(len(rows), np.nan)
    out[total > 0] = w[total > 0] @ y[pool] / total[total > 0]
    return out


# the blend is a weighted mean, so first put it back on the scale of the labels
MU, SD = oof.mean(), oof.std()
oof = y.mean() + (oof - MU) / SD * y.std()
oof_model = oof.copy()
for tr, va in folds:
    nb = neighbour_label(va, tr)
    has = ~np.isnan(nb)
    oof[va[has]] = (1 - NBR_WEIGHT) * oof[va[has]] + NBR_WEIGHT * nb[has]
print(f'Тварин у train зі схожими описами в інших фолдах: {{sum((~np.isnan(neighbour_label(va, tr))).sum() for tr, va in folds)}}')
print(f'OOF QWK лише модель: {{qwk(y, to_classes(oof_model, fit_cuts(oof_model, y))):.4f}}')
print(f'OOF QWK модель і схожі описи: {{qwk(y, to_classes(oof, fit_cuts(oof, y))):.4f}}')
''')

if CFG['twins']:
    code(r'''
def twin_label(rows, pool):
    # mean label of the twins of every row among the train rows in pool, NaN if there are none
    pool = set(pool)
    return np.array([np.mean([y[j] for j in twins[i] & pool]) if twins[i] & pool else np.nan for i in rows])


TWIN_WEIGHT = 0.5
oof_model = oof.copy()
for tr, va in folds:
    tw = twin_label(va, tr)
    has = ~np.isnan(tw)
    oof[va[has]] = (1 - TWIN_WEIGHT) * oof[va[has]] + TWIN_WEIGHT * tw[has]
print(f'OOF QWK лише модель: {qwk(y, to_classes(oof_model, fit_cuts(oof_model, y))):.4f}')
print(f'OOF QWK модель і двійники: {qwk(y, to_classes(oof, fit_cuts(oof, y))):.4f}')
''')

code(r'''
CUTS = fit_cuts(oof, y)
oof_classes = to_classes(oof, CUTS)
print(f'OOF QWK з базовими порогами: {qwk(y, to_classes(oof, np.array(CLASSES[:-1]) + 0.5)):.4f}')
print(f'OOF QWK з підібраними порогами: {qwk(y, oof_classes):.4f}')
print('Пороги:', np.round(CUTS, 3))
''' + (r'''
# shares of the classes in train, the test cuts below keep the same shares
SHARES = np.cumsum([np.mean(y == c) for c in CLASSES[:-1]])
dist_classes = np.zeros(len(y), int)
for tr, va in folds:
    dist_classes[va] = to_classes(oof[va], np.quantile(oof[va], SHARES))
print(f'OOF QWK, якщо в кожному фолді частки класів як у train: {qwk(y, dist_classes):.4f}')
''' if DIST_CUTS else '') + ('\n# species from the photos: SigLIP compares the photo vector with the texts "a photo of a dog" and "a photo of a cat".\n# Cats are adopted faster than dogs, so each species gets its own cuts with the class shares of that species in train.\nspecies_proc = AutoProcessor.from_pretrained(\'google/siglip2-so400m-patch14-224\')\nspecies_model = AutoModel.from_pretrained(\'google/siglip2-so400m-patch14-224\').eval()\nwith torch.no_grad():\n    prompts = species_proc(text=[\'a photo of a dog\', \'a photo of a cat\'], padding=\'max_length\', max_length=64, return_tensors=\'pt\')\n    text_vec = species_model.get_text_features(**prompts)\n    text_vec = text_vec.pooler_output if hasattr(text_vec, \'pooler_output\') else text_vec\n    text_vec = torch.nn.functional.normalize(text_vec, dim=-1).numpy()\nphoto_vec = raw_emb[\'so400m\'] / (np.linalg.norm(np.nan_to_num(raw_emb[\'so400m\']), axis=1, keepdims=True) + 1e-9)\nspecies_sim = np.nan_to_num(photo_vec @ text_vec.T)\nis_cat = species_sim[:, 1] > species_sim[:, 0]\nprint(f\'Котів: {is_cat.mean():.1%}\')\nfor name, group in ((\'собаки\', ~is_cat[:len(train)]), (\'коти\', is_cat[:len(train)])):\n    print(name, \'частки класів у train:\', np.round(np.bincount(y[group], minlength=max(CLASSES) + 1)[CLASSES] / group.sum(), 3))\n\n\ndef species_classes(pred, rows, pool):\n    # cuts inside each species at the class shares of that species among the train pets in pool\n    out = np.zeros(len(rows), int)\n    for cat in (False, True):\n        in_rows, in_pool = is_cat[rows] == cat, pool[is_cat[pool] == cat]\n        shares = np.cumsum([np.mean(y[in_pool] == c) for c in CLASSES[:-1]])\n        out[in_rows] = to_classes(pred[in_rows], np.quantile(pred[in_rows], shares))\n    return out\n\n\nspecies_oof = np.zeros(len(y), int)\nfor tr, va in folds:\n    species_oof[va] = species_classes(oof[va], va, tr)\nprint(f\'OOF QWK, якщо пороги окремі для котів і собак: {qwk(y, species_oof):.4f}\')\n' if SPECIES_CUTS else ''))

code(r'''
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
cm = confusion_matrix(y, oof_classes, labels=CLASSES)
axes[0].imshow(cm, cmap='Blues')
for (i, j), v in np.ndenumerate(cm):
    axes[0].text(j, i, v, ha='center', va='center')
axes[0].set_xticks(range(len(CLASSES)), CLASSES)
axes[0].set_yticks(range(len(CLASSES)), CLASSES)
axes[0].set_xlabel('Прогноз')
axes[0].set_ylabel('Фактичний клас')
axes[0].set_title('Матриця помилок (out-of-fold)')

GROUP_NAMES = {'img_eff': 'фото, EfficientNet', 'img_siglip': 'фото, SigLIP base', 'img_so400m': 'фото, SigLIP so400m',
               'img_clipL': 'фото, CLIP', 'img_dino': 'фото, DINOv2'}
importance = pd.Series(model.feature_importance('gain'), index=X.columns)
groups = importance.groupby(lambda c: GROUP_NAMES.get(c.rsplit('_', 1)[0],
                            'текст' if c.startswith(('tfidf_', 'text_emb_', 'desc_')) else 'інше')).sum()
(groups / groups.sum() * 100).sort_values().plot.barh(ax=axes[1], title='Внесок груп ознак, %')
plt.tight_layout()
plt.show()
print('Найважливіші ознаки:', ', '.join(importance.sort_values(ascending=False).index[:12]))
''')

final_blend = ' Так само перенавчаю класифікатор і Ridge-моделі та змішую прогнози з тими самими вагами.' if BLEND else ''
cuts_text = (' Пороги для test ставлю так, щоб частки класів у прогнозі були такі самі, як у train. Це захищає від зсуву прогнозів, якщо test трохи відрізняється від train.'
             if DIST_CUTS else ' Пороги беру ті, що підібрав на out-of-fold прогнозах.')
if SPECIES_CUTS:
    cuts_text = (' Пороги для test ставлю окремо для котів і собак: у кожного виду частки класів у прогнозі такі самі, як у цього виду в train. Коти в цих даних знаходять дім швидше, тому спільні пороги зсували прогнози для обох видів. Вид визначаю за фото: SigLIP порівнює вектор фото з текстами "a photo of a dog" і "a photo of a cat".')
if CFG.get('shares'):
    cuts_text = (' Пороги для test ставлю так, щоб частки класів 1, 2, 3, 4 у прогнозі були ' + ', '.join(f'{v:.0%}' for v in CFG['shares'])
                 + '. Це трохи більше крайніх класів 1 і 4, ніж у train. За результатами попередніх сабмітів QWK на LB росте швидше за CV, тому схоже, що в test крайніх класів більше.')
final_twins = final_blend + ' Для тварин з двійником у train так само змішую прогноз із міткою двійників.' if CFG['twins'] else ''
md(rf'''
## 6. Фінальна модель і submission

Як радить завдання, фінальну модель перенавчаю на всіх навчальних даних. Кількість дерев беру як середню найкращу з фолдів, збільшену на 10%, бо даних тепер більше. {cuts_text}{final_twins}
''')

code(r'''
test_raw = np.zeros(len(X_test))
for s in range(N_SEEDS):
    final_model = lgb.train({**PARAMS, 'seed': SEED + s}, lgb.Dataset(X, y),
                            num_boost_round=int(np.mean(best_iters) * 1.1))
    test_raw += final_model.predict(X_test) / N_SEEDS
''' + (r'''
test_mc = np.zeros(len(X_test))
for s in range(N_SEEDS):
    mc = lgb.train({**MC_PARAMS, 'seed': SEED + s}, lgb.Dataset(X, y_idx), num_boost_round=int(np.mean(mc_iters) * 1.1))
    test_mc += mc.predict(X_test) @ np.array(CLASSES) / N_SEEDS
test_ridge = np.mean([Ridge(alpha=alpha).fit(z[:len(X)], y).predict(z[len(X):]) for z, alpha in ridge_inputs.values()], axis=0)
''' + (r'''test_raw = WEIGHTS[0] * test_raw + WEIGHTS[1] * test_mc + WEIGHTS[2] * test_ridge
''' if not ENSEMBLE else r'''test_svr = np.mean([SVR(C=1.0, epsilon=0.3).fit(z[:len(X)], y).predict(z[len(X):]) for z in svr_inputs], axis=0)
''' + (r'''test_allmc = np.zeros(len(X_test))
for s in range(N_SEEDS):
    m = lgb.train({**MC_PARAMS, 'seed': SEED + s}, lgb.Dataset(X_all, y_idx), num_boost_round=int(np.mean(allmc_iters) * 1.1))
    test_allmc += m.predict(X_all_test) @ np.array(CLASSES) / N_SEEDS
''' if ALLMC else '') + r'''test_models = {'регресія': test_raw, 'класифікатор': test_mc, 'Ridge': test_ridge, 'SVR': test_svr''' + (r''',
               'класифікатор, усі фото-моделі': test_allmc}''' if ALLMC else '}') + r'''
''' + (r'''test_raw = sum(WEIGHTS[k] * test_models[k] for k in WEIGHTS)
''' if not STACK else r'''''' + ('test_models.update(more_test)' + chr(10) if MEMBERS10 else '') + FT('''# the second level learned on predictions of fold models, so on test it gets the same kind: mean of the 5 fold models
test_models = {'регресія': fold_test_lgb, 'класифікатор': fold_test_mc, 'Ridge': fold_test_ridge, 'SVR': fold_test_svr,
               'класифікатор, усі фото-моделі': fold_test_allmc}
''') + r'''test_level2 = pd.concat([pd.DataFrame(test_models).rename(columns=L2_NAMES),
                         neighbour_features(np.arange(len(train), len(full)), np.arange(len(train)))], axis=1)
''' + ("test_level2['n_photos'] = test['n_photos'].values" + chr(10) if not NO_NPHOTOS else '') + r'''''' + (r'''test_level2 = pd.concat([test_level2, photo_stats.iloc[len(train):].reset_index(drop=True)], axis=1)
''' if PP_AGG else '') + r'''''' + (r'''level2_model = lgb.train(L2_PARAMS, lgb.Dataset(level2, y), num_boost_round=int(np.mean(l2_iters) * 1.1))
test_raw = level2_model.predict(test_level2)''' if L2_SEEDS == 1 else r'''test_raw = np.mean([lgb.train({**L2_PARAMS, 'seed': s}, lgb.Dataset(level2, y), num_boost_round=int(np.mean(l2_iters) * 1.1))
                    .predict(test_level2) for s in range(L2_SEEDS)], axis=0)''') + r'''
''')) + (r'''
test_raw = y.mean() + (test_raw - MU) / SD * y.std()
nb = neighbour_label(np.arange(len(train), len(full)), np.arange(len(train)))
has = ~np.isnan(nb)
test_raw[has] = (1 - NBR_WEIGHT) * test_raw[has] + NBR_WEIGHT * nb[has]
print('Тварин у test зі схожими описами в train:', has.sum())
''' if NBR else '') if BLEND else '') + (r'''
tw = twin_label(range(len(train), len(full)), range(len(train)))
has = ~np.isnan(tw)
test_raw[has] = (1 - TWIN_WEIGHT) * test_raw[has] + TWIN_WEIGHT * tw[has]
print('Тварин у test, для яких враховано двійників:', has.sum())
''' if CFG['twins'] else '') + r'''
''' + (r'''
test_pred = species_classes(test_raw, np.arange(len(train), len(full)), np.arange(len(train)))
''' if SPECIES_CUTS else (r'''
TEST_CUTS = np.quantile(test_raw, ''' + ('SHARES)' if not CFG.get('shares') else
    f"np.cumsum({CFG['shares'][:-1]!r}))  # more 1 and 4 than in train, see the text above") + r'''
print('Пороги для test:', np.round(TEST_CUTS, 3))
test_pred = to_classes(test_raw, TEST_CUTS)
''' if DIST_CUTS else r'''
test_pred = to_classes(test_raw, CUTS)
''')) + r'''submission = pd.DataFrame({ID_COL: test[ID_COL], TARGET_COL: test_pred.astype(int)})
assert len(submission) == len(test) and submission[ID_COL].is_unique
submission.to_csv('submission.csv', index=False)
np.save('test_raw.npy', test_raw)  # raw scores before the cuts, to compare versions
np.save('oof.npy', oof)
print(submission[TARGET_COL].value_counts().sort_index())
submission.head()
''')

conclusion_twins = ('\n- У даних є тварини-двійники з однаковим фото або описом. Для тварин з двійником у train прогноз змішано з міткою двійників.'
                    if CFG['twins'] else '')
conclusion_blend = ('\n- Підсумковий прогноз це суміш трьох моделей: LightGBM-регресії, LightGBM-класифікатора і Ridge-моделей на повних векторах.'
                     if BLEND else '')
conclusion_twins = conclusion_blend + conclusion_twins
conclusion_seeds = f' Прогноз усереднено за {CFG["seeds"]} seed.' if CFG['seeds'] > 1 else ''
improve = ['підібрати параметри LightGBM за допомогою Optuna;', 'донавчити CNN на фото тварин замість готових ембедингів;']
if 'so400m' not in CFG['images']:
    improve.append('спробувати більші попередньо навчені моделі для фото, наприклад більшу версію SigLIP;')
if CFG['seeds'] == 1:
    improve.append('усереднити кілька моделей з різними seed.')
else:
    improve.append('використати всі фото тварини, а не лише перші три.')
improve[-1] = improve[-1].rstrip(';').rstrip('.') + '.'
improve_text = '\n'.join(f'- {s}' for s in improve)
md(rf'''
## 7. Висновки

- Модель поєднує текст опису (TF-IDF і ембединги трансформера) і фото ({images_list}).{conclusion_twins}
- Якість я перевіряю {CFG.get('n_folds', 5)}-кратною крос-валідацією за метрикою змагання QWK. Пороги класів підібрано на out-of-fold прогнозах. Результати, матриця помилок і внесок груп ознак видно у виводах розділу 5.
- Фінальну модель перенавчено на всіх навчальних даних, прогноз збережено в submission.csv.{conclusion_seeds}

Що можна покращити:
{improve_text}
''')

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
nbf.write(nb, OUT)
print('written', OUT, len(cells), 'cells')
