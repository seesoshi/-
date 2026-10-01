**Сравнение четырёх типов контентных сигналов в рекомендательной системе**

_текстовый · аудио · графовый (#nowplaying-RS) · мультимодальный (VK-LSVD)_


# **Введение**

В основной части проекта (датасет VK-LSVD) мы выяснили, что контентный эмбеддинг улучшает рекомендации коллаборативной фильтрации, — но там был доступен только один, мультимодальный эмбеддинг, и сравнить между собой разные типы сигналов было не на чем. В этой части мы восполняем пробел на другом открытом датасете, где четыре типа сигналов собираются честно для одного и того же каталога объектов.

Датасет — **#nowplaying-RS** (Zenodo, лицензия CC BY): 11,6 млн событий прослушивания музыки, собранных из Twitter. Что делает его подходящим именно для нашей задачи: (1) к каждому треку прилагаются **готовые аудио-характеристики Spotify** — danceability, energy, tempo и другие, то есть аудио-сигнал не нужно извлекать самим; (2) слушатели помечали треки **хэштегами** (#rock, #chill, #90s) — это готовый текстовый сигнал социальных тегов; (3) **графовый** сигнал мы обучаем сами из графа взаимодействий методом item2vec; (4) **мультимодальный** — берём настоящий готовый: эмбеддинг VK-LSVD из основной части проекта, обученный одной моделью сразу на кадрах, аудио и описании видео. Каталоги двух датасетов разные, поэтому абсолютные метрики между ними не сравниваются — сравнивается нормированная величина: прирост гибрида к собственному ALS-baseline (в процентах) и его аналог на хвосте непопулярных объектов.


# **Цель работы**

Сравнить влияние четырёх типов контентных сигналов (текстового, аудио, графового, мультимодального) на качество рекомендаций: как самостоятельных моделей и как добавок к коллаборативной фильтрации — и определить, какой тип сигнала полезнее в каких сценариях.


# **Задачи работы**

1\. Загрузить датасет #nowplaying-RS, отобрать активных пользователей и треки, выполнить разбиение по времени. 2. Построить матрицу взаимодействий и обучить baseline — ALS. 3. Сконструировать четыре сигнальных матрицы: аудио (из готовых характеристик), текст (TF-IDF + SVD по хэштегам), граф (item2vec), мультимодальный (конкатенация). 4. Для каждого сигнала оценить контент-модель соло и гибрид с ALS (с подбором α), включая сегмент малопопулярных треков. 5. Независимо оценить информативность сигналов методом permutation importance. 6. Сформулировать выводы о полезности типов сигналов по сценариям.


# **Ход работы**

## **Шаг 1. Установка и загрузка данных**

Ставим библиотеки и скачиваем два файла датасета с Zenodo: таблицу событий (кто, какой трек, какой хэштег, когда) и таблицу аудио-характеристик. Фиксируем зерно случайности — без этого результаты не воспроизвести.

\# ========== 0. Установка и загрузка данных ==========

\# !pip install implicit polars scikit-learn gensim

\# Файлы датасета (скачать один раз с Zenodo: https\://zenodo.org/record/3247476):

\# user\_track\_hashtag\_timestamp.csv - события: user\_id, track\_id, hashtag, created\_at

\# context\_content\_features.csv - аудио-характеристики Spotify по каждому событию

\# !wget https\://zenodo.org/record/3247476/files/user\_track\_hashtag\_timestamp.csv

\# !wget https\://zenodo.org/record/3247476/files/context\_content\_features.csv

import polars as pl

import numpy as np

import random

from scipy.sparse import coo\_matrix

from implicit.als import AlternatingLeastSquares

RANDOM\_SEED = 42

random.seed(RANDOM\_SEED); np.random.seed(RANDOM\_SEED)


## **Шаг 2. Читаем данные и отбираем активных**

Полный датасет велик для учебной машины, поэтому оставляем пользователей с ≥20 событиями и треки с ≥10 — это тот же приём, что сабсэмплы «активных» в VK-LSVD: на слишком редких пользователях и треках любой модели просто не на чем учиться. Печать колонок в начале — привычка самопроверки: имена в CSV стоит сверить с документацией датасета до того, как что-то падает дальше.

\# ========== 1. Читаем события и аудио-фичи ==========

events = pl.read\_csv('user\_track\_hashtag\_timestamp.csv',

columns=\['user\_id', 'track\_id', 'hashtag', 'created\_at'])

print('events columns:', events.columns, '| строк:', len(events))

AUDIO\_COLS = \['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',

'instrumentalness', 'liveness', 'valence', 'tempo']

ctx = pl.read\_csv('context\_content\_features.csv',

columns=\['track\_id'] + AUDIO\_COLS)

print('ctx columns:', ctx.columns)

\# --- сабсэмплинг: активные пользователи и треки (плотность как у up/ip-срезов VK-LSVD) ---

MIN\_USER\_EVENTS, MIN\_TRACK\_EVENTS = 20, 10

u\_cnt = events.group\_by('user\_id').len()

t\_cnt = events.group\_by('track\_id').len()

active\_users = u\_cnt.filter(pl.col('len') >= MIN\_USER\_EVENTS)\['user\_id']

active\_tracks = t\_cnt.filter(pl.col('len') >= MIN\_TRACK\_EVENTS)\['track\_id']

events = events.filter(pl.col('user\_id').is\_in(active\_users) &

pl.col('track\_id').is\_in(active\_tracks))

print('после фильтра:', len(events), 'событий,',

events\['user\_id'].n\_unique(), 'пользователей,',

events\['track\_id'].n\_unique(), 'треков')


## **Шаг 3. Разбиение по времени и матрица взаимодействий**

Чтобы модель не «подглядывала в будущее», делим события по времени: первые 90% — обучение, последние 10% — проверка (temporal split, как и в основной части). Вес взаимодействия здесь проще, чем в VK-LSVD: сигнал один — сам факт прослушивания, поэтому вес = 1 + log(1 + число прослушиваний пары «пользователь–трек»). Логарифм — та же форма уверенности из статьи WRMF: десятое прослушивание говорит об интересе меньше, чем второе. Матрица собирается строго «пользователи × треки» — с assert-проверкой размерностей (урок главного бага основной части).

\# ========== 2. Split по времени + матрица users x tracks ==========

events = events.sort('created\_at')

split\_ts = events\['created\_at'].quantile(0.9) # первые 90% времени - train

train\_ev = events.filter(pl.col('created\_at') < split\_ts)

val\_ev = events.filter(pl.col('created\_at') >= split\_ts)

print(f'train: {len(train\_ev)}, validation: {len(val\_ev)}')

\# вес = log1p(количество прослушиваний пары user-track в train)

train\_w = (train\_ev.group\_by(\['user\_id', 'track\_id']).len()

.with\_columns((1.0 + pl.col('len').log1p()).alias('weight')))

user\_ids = sorted(train\_w\['user\_id'].unique().to\_list())

item\_ids = sorted(train\_w\['track\_id'].unique().to\_list())

user\_to\_idx = {u: i for i, u in enumerate(user\_ids)}

item\_to\_idx = {t: i for i, t in enumerate(item\_ids)}

rows = train\_w\['user\_id'].replace(user\_to\_idx).to\_numpy()

cols = train\_w\['track\_id'].replace(item\_to\_idx).to\_numpy()

data = train\_w\['weight'].to\_numpy()

sparse\_matrix = coo\_matrix((data, (rows, cols)),

shape=(len(user\_ids), len(item\_ids))).tocsr()

sparse\_matrix.eliminate\_zeros()

assert sparse\_matrix.shape == (len(user\_ids), len(item\_ids))

print('Matrix:', sparse\_matrix.shape, 'nnz:', sparse\_matrix.nnz)

val\_interactions = val\_ev.select(\['user\_id', 'track\_id']).rename({'track\_id': 'item\_id'}).unique()


## **Шаг 4. Строим четыре сигнала**

**Аудио — уже готов. **Характеристики Spotify лежат по каждому событию; усредняем по треку и стандартизуем (z-score), иначе tempo со значениями \~120 задавил бы danceability со значениями \~0.6 при вычислении косинуса.

\# ========== 3.1 АУДИО-сигнал: готовые характеристики Spotify ==========

audio\_by\_track = (ctx.filter(pl.col('track\_id').is\_in(item\_ids))

.group\_by('track\_id').mean())

audio\_map = {r\['track\_id']: \[r\[c] for c in AUDIO\_COLS]

for r in audio\_by\_track.iter\_rows(named=True)}

audio\_matrix = np.zeros((len(item\_ids), len(AUDIO\_COLS)), dtype=np.float32)

for iid, idx in item\_to\_idx.items():

if iid in audio\_map:

audio\_matrix\[idx] = audio\_map\[iid]

\# z-score стандартизация по столбцам

audio\_matrix = (audio\_matrix - audio\_matrix.mean(0)) / (audio\_matrix.std(0) + 1e-9)

print('audio\_matrix:', audio\_matrix.shape)

**Текст — из хэштегов. **Описаний у треков нет, но есть социальные теги. Склеиваем хэштеги трека в «документ» и строим классический латентно-семантический вектор: TF-IDF (насколько тег характерен именно для этого трека) → усечённый SVD до 64 компонент (сжатие в плотный вектор). Никаких тяжёлых нейросетей — метод понятен и работает.

\# ========== 3.2 ТЕКСТ-сигнал: хэштеги -> TF-IDF -> SVD ==========

from sklearn.feature\_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD

tag\_docs = (train\_ev.filter(pl.col('hashtag').is\_not\_null())

.group\_by('track\_id')

.agg(pl.col('hashtag').str.concat(' ').alias('doc')))

doc\_map = dict(zip(tag\_docs\['track\_id'].to\_list(), tag\_docs\['doc'].to\_list()))

corpus = \[doc\_map.get(iid, '') for iid in item\_ids] # строго в порядке item\_to\_idx

tfidf = TfidfVectorizer(max\_features=5000, min\_df=3)

X\_tfidf = tfidf.fit\_transform(corpus)

text\_matrix = TruncatedSVD(n\_components=64, random\_state=RANDOM\_SEED)\\

.fit\_transform(X\_tfidf).astype(np.float32)

print('text\_matrix:', text\_matrix.shape)

**Граф — обучаем сами. **Графовый эмбеддинг кодирует соседство в графе «пользователь–трек»: треки, которые слушают одни и те же люди подряд, получают близкие векторы. Делаем упрощённый DeepWalk, который называют item2vec: «предложение» = последовательность треков одного пользователя по времени, и по таким предложениям обучаем обычный word2vec. Важная честная оговорка: этот сигнал — поведенческий, а не контентный; в сравнении он отвечает на вопрос «а что если вместо содержания взять сжатое поведение?».

\# ========== 3.3 ГРАФ-сигнал: item2vec (word2vec по сессиям) ==========

from gensim.models import Word2Vec

sentences = (train\_ev.sort('created\_at')

.group\_by('user\_id')

.agg(pl.col('track\_id').cast(pl.Utf8).alias('seq'))

)\['seq'].to\_list()

w2v = Word2Vec(sentences=sentences, vector\_size=64, window=5,

min\_count=1, sg=1, epochs=5, seed=RANDOM\_SEED, workers=4)

graph\_matrix = np.zeros((len(item\_ids), 64), dtype=np.float32)

for iid, idx in item\_to\_idx.items():

key = str(iid)

if key in w2v.wv:

graph\_matrix\[idx] = w2v.wv\[key]

print('graph\_matrix:', graph\_matrix.shape)

**Мультимодальный — готовый эмбеддинг VK-LSVD. **Склейка текста с аудио была бы лишь суррогатом мультимодальности, поэтому берём настоящий готовый мультимодальный эмбеддинг из VK-LSVD: он обучен одной моделью сразу на кадрах, аудио и описании видео, строго на контенте, с компонентами, упорядоченными по значимости. Код ниже — компактный мини-пайплайн VK-LSVD: скачиваем сабсэмпл, собираем матрицу той же log-схемой весов с фильтром нулей и готовим матрицу эмбеддингов (32 старшие компоненты).

\# ========== 3.4 МУЛЬТИМОДАЛЬНЫЙ сигнал: готовый эмбеддинг VK-LSVD ==========

from huggingface\_hub import hf\_hub\_download

SUB = 'up0.001\_ip0.001'

vk\_files = (\[f'subsamples/{SUB}/train/week\_{i:02}.parquet' for i in range(25)]

\+ \[f'subsamples/{SUB}/validation/week\_25.parquet', 'metadata/item\_embeddings.npz'])

for f\_ in vk\_files:

hf\_hub\_download(repo\_id='deepvk/VK-LSVD', repo\_type='dataset', filename=f\_, local\_dir='VK-LSVD')

vk\_train = pl.concat(\[pl.scan\_parquet(f'VK-LSVD/subsamples/{SUB}/train/week\_{i:02}.parquet')

for i in range(25)]).collect(engine='streaming')

vk\_val\_interactions = (pl.read\_parquet(f'VK-LSVD/subsamples/{SUB}/validation/week\_25.parquet')

.select(\['user\_id', 'item\_id']).unique())

emb\_npz = np.load('VK-LSVD/metadata/item\_embeddings.npz')

emb\_pos = {int(t): k for k, t in enumerate(emb\_npz\['item\_id'])}

vk\_items = sorted(set(vk\_train\['item\_id'].unique().to\_list()) & set(emb\_pos.keys()))

vk\_users = sorted(vk\_train\['user\_id'].unique().to\_list())

vk\_user\_to\_idx = {u: i for i, u in enumerate(vk\_users)}

vk\_item\_to\_idx = {t: i for i, t in enumerate(vk\_items)}

\# матрица эмбеддингов (мультимодальный сигнал), 32 старшие компоненты

multi\_matrix = np.stack(\[emb\_npz\['embedding']\[emb\_pos\[t]]\[:32]

for t in vk\_items]).astype(np.float32)

\# веса как в основной части: log-время + бонусы действий, клип, фильтр нулей

vk\_w = (vk\_train.filter(pl.col('item\_id').is\_in(vk\_items))

.with\_columns((1.0 + ((pl.col('timespent') / 60).clip(0, 60) + 1.0).log()

\+ pl.col('like').cast(pl.Int32) \* 3.0

\+ pl.col('share').cast(pl.Int32) \* 2.5

\+ pl.col('bookmark').cast(pl.Int32) \* 2.0

\+ pl.col('click\_on\_author').cast(pl.Int32) \* 1.5

\+ pl.col('open\_comments').cast(pl.Int32) \* 1.0

\+ pl.col('dislike').cast(pl.Int32) \* (-2.0)

).clip(0, 100).alias('weight'))

.filter(pl.col('weight') > 0))

r\_ = vk\_w\['user\_id'].replace(vk\_user\_to\_idx).to\_numpy()

c\_ = vk\_w\['item\_id'].replace(vk\_item\_to\_idx).to\_numpy()

vk\_matrix = coo\_matrix((vk\_w\['weight'].to\_numpy(), (r\_, c\_)),

shape=(len(vk\_users), len(vk\_items))).tocsr()

vk\_matrix.eliminate\_zeros()

print('VK-LSVD matrix:', vk\_matrix.shape, 'nnz:', vk\_matrix.nnz)

print('multi\_matrix (VK-LSVD embedding):', multi\_matrix.shape)

signal\_matrices = {'text': text\_matrix, 'audio': audio\_matrix, 'graph': graph\_matrix}


## **Шаг 5. Baseline и универсальная контент-модель**

Начинаем с простой модели, встроенной в модуль implicit, — ALS: она выучивает по 64-мерному вектору на пользователя и трек так, чтобы их скалярное произведение воспроизводило вес прослушивания. Функции оценки (Recall\@10, NDCG\@10) переносим из основной части без изменений — включая исправление NDCG (ноль записывается каждому пользователю без попаданий).

\# ========== 4.2 ALS-baseline ==========

model = AlternatingLeastSquares(factors=64, regularization=0.1,

iterations=20, alpha=1.0, random\_state=RANDOM\_SEED)

model.fit(sparse\_matrix)

def als\_scores(ui):

return model.item\_factors @ model.user\_factors\[ui]

print('ALS:', evaluate\_score\_fn(als\_scores, val\_interactions, sparse\_matrix, user\_to\_idx, item\_to\_idx))

Дальше — главная выгода архитектуры: контент-модель и гибрид написаны один раз для ЛЮБОГО сигнала. Функция make\_content\_scores принимает любую матрицу «трек × признаки», строит профили пользователей (взвешенное среднее векторов прослушанного) и возвращает косинусный скорер; make\_hybrid смешивает её с ALS через α после min-max нормализации обеих шкал.

\# ========== 4.3 Контент-модель и гибрид для ЛЮБОГО сигнала и ЛЮБОГО датасета ==========

def make\_content\_scores(content\_matrix, inter\_matrix):

ws = inter\_matrix @ content\_matrix

wt = np.asarray(inter\_matrix.sum(axis=1)).flatten(); wt\[wt == 0] = 1.0

up = ws / wt\[:, None]

cn = np.linalg.norm(content\_matrix, axis=1) + 1e-9

def cs(ui, cm=content\_matrix, up=up, cn=cn):

p = up\[ui]

return (cm @ p) / (cn \* (np.linalg.norm(p) + 1e-9))

return cs

def min\_max(x):

lo, hi = x.min(), x.max()

return (x - lo) / (hi - lo) if hi > lo else np.zeros\_like(x)

def make\_hybrid(cf\_fn, content\_fn, alpha):

def h(ui):

return alpha \* min\_max(cf\_fn(ui)) + (1 - alpha) \* min\_max(content\_fn(ui))

return h


## **Шаг 6. Главный эксперимент — сравнение сигналов**

Универсальная функция run\_signal делает для каждого сигнала одно и то же: (а) контент-модель соло; (б) лучший гибрид с ALS (α по сетке 0.3–0.9 — разным сигналам ALS «доверяет» по-разному); (в) прирост гибрида к baseline на всём каталоге и отдельно на «хвосте» (80% наименее популярных объектов — сценарий холодного старта). Три сигнала прогоняются на nowplaying-RS; мультимодальный — на VK-LSVD со своим собственным ALS-baseline. Сравнение между датасетами ведётся по нормированным колонкам «прирост» и «прирост tail».

\# ========== 5. Сравнение сигналов (nowplaying-RS + мультимодальный на VK-LSVD) ==========

def run\_signal(name, content\_matrix, inter\_matrix, val\_inter, u2i, i2i, cf\_fn, base\_recall):

cs = make\_content\_scores(content\_matrix, inter\_matrix)

solo = evaluate\_score\_fn(cs, val\_inter, inter\_matrix, u2i, i2i)

best = (0, None)

for alpha in \[0.3, 0.5, 0.7, 0.9]:

m = evaluate\_score\_fn(make\_hybrid(cf\_fn, cs, alpha), val\_inter, inter\_matrix, u2i, i2i)

if m\['Recall\@K'] > best\[0]:

best = (m\['Recall\@K'], alpha)

pop = np.asarray(inter\_matrix.sum(axis=0)).flatten()

tail = set(np.argsort(-pop)\[int(0.2 \* inter\_matrix.shape\[1]):].tolist())

h\_tail = evaluate\_score\_fn(make\_hybrid(cf\_fn, cs, best\[1]), val\_inter, inter\_matrix, u2i, i2i,

item\_idx\_filter=tail)

b\_tail = evaluate\_score\_fn(cf\_fn, val\_inter, inter\_matrix, u2i, i2i, item\_idx\_filter=tail)

gain = (best\[0] / base\_recall - 1) \* 100

gain\_tail = ((h\_tail\['Recall\@K'] / b\_tail\['Recall\@K'] - 1) \* 100

if b\_tail\['Recall\@K'] > 0 else float('nan'))

print(f"{name:<14} {solo\['Recall\@K']:>8.4f} {best\[0]:>8.4f} {best\[1]:>6} "

f"{gain:>+9.1f}% {gain\_tail:>+11.1f}%")

return dict(solo=solo\['Recall\@K'], hybrid=best\[0], alpha=best\[1],

gain=gain, gain\_tail=gain\_tail)

\# baseline nowplaying-RS

base\_np = evaluate\_score\_fn(als\_scores, val\_interactions, sparse\_matrix, user\_to\_idx, item\_to\_idx)

print(f"ALS baseline (nowplaying): Recall={base\_np\['Recall\@K']:.4f}")

print(f"{'Сигнал':<14} {'Solo':>8} {'Hybrid':>8} {'alpha':>6} {'прирост':>10} {'прирост tail':>12}")

results = {}

for name, cm in signal\_matrices.items():

results\[name] = run\_signal(name, cm, sparse\_matrix, val\_interactions,

user\_to\_idx, item\_to\_idx, als\_scores, base\_np\['Recall\@K'])

\# мультимодальный — на VK-LSVD: свой ALS и свой baseline

vk\_als = AlternatingLeastSquares(factors=64, regularization=0.1,

iterations=20, alpha=1.0, random\_state=RANDOM\_SEED)

vk\_als.fit(vk\_matrix)

def vk\_als\_scores(ui):

return vk\_als.item\_factors @ vk\_als.user\_factors\[ui]

base\_vk = evaluate\_score\_fn(vk\_als\_scores, vk\_val\_interactions, vk\_matrix,

vk\_user\_to\_idx, vk\_item\_to\_idx)

print(f"ALS baseline (VK-LSVD): Recall={base\_vk\['Recall\@K']:.4f}")

results\['multimodal(VK)'] = run\_signal('multimod.(VK)', multi\_matrix, vk\_matrix,

vk\_val\_interactions, vk\_user\_to\_idx, vk\_item\_to\_idx,

vk\_als\_scores, base\_vk\['Recall\@K'])

\# Сигналы сравниваем по колонкам "прирост" и "прирост tail": они нормированы

\# на собственный baseline и потому сопоставимы между датасетами.


## **Шаг 7. Независимая проверка — важность сигналов**

Второй, независимый способ измерить информативность: бустинг учится предсказывать вес прослушивания сразу из нескольких сигналов (по 8 первых компонент каждого — блоки равного размера), затем мы по очереди перемешиваем каждый блок и смотрим, без какого предсказание рушится сильнее. Метод работает только для сигналов одного каталога, поэтому здесь три блока nowplaying-RS (text/audio/graph); информативность мультимодального сигнала VK-LSVD показана в шаге 6 через прирост к собственному baseline (+21% к ALS в основной части). Если ранжирование блоков совпадёт с итогами шага 6 — выводы подтверждены двумя методиками.

\# ========== 6. Permutation importance по блокам сигналов ==========

from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.inspection import permutation\_importance

from sklearn.model\_selection import train\_test\_split

N\_COMP, N\_SAMPLE = 8, 200\_000

samp = train\_w\.sample(n=min(N\_SAMPLE, len(train\_w)), seed=RANDOM\_SEED)

iidx = samp\['track\_id'].replace(item\_to\_idx).to\_numpy()

blocks, X\_parts, feat\_names = {}, \[], \[]

for name, cm in signal\_matrices.items():

n = min(N\_COMP, cm.shape\[1])

X\_parts.append(cm\[iidx, :n])

cols = \[f'{name}\_{i}' for i in range(n)]

blocks\[name] = cols; feat\_names += cols

X = np.hstack(X\_parts).astype(np.float32)

y = np.log1p(samp\['weight'].to\_numpy())

Xtr, Xte, ytr, yte = train\_test\_split(X, y, test\_size=0.2, random\_state=RANDOM\_SEED)

gbm = HistGradientBoostingRegressor(max\_iter=150, random\_state=RANDOM\_SEED).fit(Xtr, ytr)

print(f'R^2: {gbm.score(Xte, yte):.3f}')

pi = permutation\_importance(gbm, Xte\[:20000], yte\[:20000], n\_repeats=3, random\_state=RANDOM\_SEED)

imp = dict(zip(feat\_names, pi.importances\_mean))

print('\n--- Важность по блокам ---')

for b, cols\_ in blocks.items():

print(f'{b:<12} {sum(imp\[c] for c in cols\_):.4f}')


# **Результаты**

_Таблица заполняется после прогона; структура колонок соответствует печати шага 6._

|                 |                   |          |            |         |                   |                  |
| --------------- | ----------------- | -------- | ---------- | ------- | ----------------- | ---------------- |
| **Сигнал**      | **Датасет**       | **Solo** | **Hybrid** | **α\*** | **Прирост к ALS** | **Прирост tail** |
|                 |                   |          |            |         |                   |                  |
|                 |                   |          |            |         |                   |                  |
|                 |                   |          |            |         |                   |                  |
|                 |                   |          |            |         |                   |                  |
|                 |                   |          |            |         |                   |                  |
|                 |                   |          |            |         |                   |                  |
|                 |                   |          |            |         |                   |                  |
|                 |                   |          |            |         |                   |                  |
|                 |                   |          |            |         |                   |                  |
| Текстовый       | nowplaying-RS     | \[...]   | \[...]     | \[...]  | \[...]            | \[...]           |
|                 |                   |          |            |         |                   |                  |
|                 |                   |          |            |         |                   |                  |
|                 |                   |          |            |         |                   |                  |
|                 |                   |          |            |         |                   |                  |
|                 |                   |          |            |         |                   |                  |
|                 |                   |          |            |         |                   |                  |
|                 |                   |          |            |         |                   |                  |
|                 |                   |          |            |         |                   |                  |
|                 |                   |          |            |         |                   |                  |
| Аудио           | nowplaying-RS     | \[...]   | \[...]     | \[...]  | \[...]            | \[...]           |
|                 |                   |          |            |         |                   |                  |
|                 |                   |          |            |         |                   |                  |
|                 |                   |          |            |         |                   |                  |
|                 |                   |          |            |         |                   |                  |
|                 |                   |          |            |         |                   |                  |
|                 |                   |          |            |         |                   |                  |
|                 |                   |          |            |         |                   |                  |
|                 |                   |          |            |         |                   |                  |
|                 |                   |          |            |         |                   |                  |
| Графовый        | nowplaying-RS     | \[...]   | \[...]     | \[...]  | \[...]            | \[...]           |
|                 |                   |          |            |         |                   |                  |
|                 |                   |          |            |         |                   |                  |
|                 |                   |          |            |         |                   |                  |
|                 |                   |          |            |         |                   |                  |
|                 |                   |          |            |         |                   |                  |
|                 |                   |          |            |         |                   |                  |
|                 |                   |          |            |         |                   |                  |
|                 |                   |          |            |         |                   |                  |
|                 |                   |          |            |         |                   |                  |
| Мультимодальный | VK-LSVD (готовый) | 0.0026   | 0.0208     | 0.5     | +21%              | \[...]           |

Как читать: **Solo и Hybrid** сопоставимы только внутри одного датасета; между датасетами сравниваются **нормированные колонки** «Прирост к ALS» и «Прирост tail» — они посчитаны относительно собственного baseline каждого датасета. В строке мультимодального уже проставлены цифры основной части (гибрид ALS+Content на VK-LSVD, α=0.5); его прирост на хвосте заполняется финальным прогоном.


# **Выводы (акцент — типы сигналов)**

**1. Типы контентных сигналов не равнозначны, и их полезность зависит от роли. **Сигнал, сильный «соло», не обязательно лучшая добавка к CF, и наоборот: колонки Solo и Прирост в таблице отвечают на два разных вопроса. Для задания важен второй — прирост к коллаборативной фильтрации.

**2. Графовый сигнал — сильный соло, но слабая добавка (ожидание для проверки цифрами). **Item2vec выучен из того же поведения, что и ALS, поэтому в одиночку он близок к CF по силе, но в гибриде почти не добавляет нового: два поведенческих сигнала дублируют друг друга. Урок о типах: ценность добавки определяется не силой, а НЕзависимостью от базовой модели.

**3. Текстовый и аудио-сигналы — слабые соло, но комплементарные (ожидание для проверки). **Они описывают треки с той стороны, которую поведение не видит («о чём» и «как звучит»), поэтому ожидаемо дают заметный прирост в гибриде и особенно на хвосте непопулярных треков, где у CF мало данных. Это повторяет главный вывод основной части на VK-LSVD — комплементарность важнее самостоятельной силы.

**4. Мультимодальный сигнал — эталон среди контентных, и у него уже есть реальная цифра. **Готовый мультимодальный эмбеддинг VK-LSVD (кадры+аудио+описание, обученные совместно) дал +21% Recall\@10 к ALS в основной части проекта — это планка для одиночных модальностей: если текст или аудио дадут сопоставимый нормированный прирост на nowplaying-RS, одной модальности почти достаточно; если заметно меньший — совместное обучение модальностей действительно добавляет ценность.

**5. Сценарии. **По совокупности с основной частью: контентные сигналы (текст/аудио/мультимодальный) наиболее полезны (а) для CF-моделей средней силы, (б) на малопопулярных объектах (холодный старт), (в) для разнообразия выдачи; поведенческо-графовые сигналы полезны там, где CF-модель недоступна или слишком дорога, но как добавка к ней избыточны.

**Ограничения. **Текстовый сигнал построен на социальных тегах, а не на описаниях — он «шумнее» редакционного текста; аудио-сигнал — 9 обобщённых характеристик, а не полноценный нейросетевой эмбеддинг звука; мультимодальный сигнал измерен на другом каталоге (VK-LSVD, видео) и сравнивается с остальными только через нормированный прирост к собственному baseline — различие доменов (музыка/видео) остаётся неустранимым фактором и оговаривается при интерпретации.


# **Приложение. Полный код**

\# ========== 0. Установка и загрузка данных ==========

\# !pip install implicit polars scikit-learn gensim

\# Файлы датасета (скачать один раз с Zenodo: https\://zenodo.org/record/3247476):

\# user\_track\_hashtag\_timestamp.csv - события: user\_id, track\_id, hashtag, created\_at

\# context\_content\_features.csv - аудио-характеристики Spotify по каждому событию

\# !wget https\://zenodo.org/record/3247476/files/user\_track\_hashtag\_timestamp.csv

\# !wget https\://zenodo.org/record/3247476/files/context\_content\_features.csv

import polars as pl

import numpy as np

import random

from scipy.sparse import coo\_matrix

from implicit.als import AlternatingLeastSquares

RANDOM\_SEED = 42

random.seed(RANDOM\_SEED); np.random.seed(RANDOM\_SEED)

\# ========== 1. Читаем события и аудио-фичи ==========

events = pl.read\_csv('user\_track\_hashtag\_timestamp.csv',

columns=\['user\_id', 'track\_id', 'hashtag', 'created\_at'])

print('events columns:', events.columns, '| строк:', len(events))

AUDIO\_COLS = \['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',

'instrumentalness', 'liveness', 'valence', 'tempo']

ctx = pl.read\_csv('context\_content\_features.csv',

columns=\['track\_id'] + AUDIO\_COLS)

print('ctx columns:', ctx.columns)

\# --- сабсэмплинг: активные пользователи и треки (плотность как у up/ip-срезов VK-LSVD) ---

MIN\_USER\_EVENTS, MIN\_TRACK\_EVENTS = 20, 10

u\_cnt = events.group\_by('user\_id').len()

t\_cnt = events.group\_by('track\_id').len()

active\_users = u\_cnt.filter(pl.col('len') >= MIN\_USER\_EVENTS)\['user\_id']

active\_tracks = t\_cnt.filter(pl.col('len') >= MIN\_TRACK\_EVENTS)\['track\_id']

events = events.filter(pl.col('user\_id').is\_in(active\_users) &

pl.col('track\_id').is\_in(active\_tracks))

print('после фильтра:', len(events), 'событий,',

events\['user\_id'].n\_unique(), 'пользователей,',

events\['track\_id'].n\_unique(), 'треков')

\# ========== 2. Split по времени + матрица users x tracks ==========

events = events.sort('created\_at')

split\_ts = events\['created\_at'].quantile(0.9) # первые 90% времени - train

train\_ev = events.filter(pl.col('created\_at') < split\_ts)

val\_ev = events.filter(pl.col('created\_at') >= split\_ts)

print(f'train: {len(train\_ev)}, validation: {len(val\_ev)}')

\# вес = log1p(количество прослушиваний пары user-track в train)

train\_w = (train\_ev.group\_by(\['user\_id', 'track\_id']).len()

.with\_columns((1.0 + pl.col('len').log1p()).alias('weight')))

user\_ids = sorted(train\_w\['user\_id'].unique().to\_list())

item\_ids = sorted(train\_w\['track\_id'].unique().to\_list())

user\_to\_idx = {u: i for i, u in enumerate(user\_ids)}

item\_to\_idx = {t: i for i, t in enumerate(item\_ids)}

rows = train\_w\['user\_id'].replace(user\_to\_idx).to\_numpy()

cols = train\_w\['track\_id'].replace(item\_to\_idx).to\_numpy()

data = train\_w\['weight'].to\_numpy()

sparse\_matrix = coo\_matrix((data, (rows, cols)),

shape=(len(user\_ids), len(item\_ids))).tocsr()

sparse\_matrix.eliminate\_zeros()

assert sparse\_matrix.shape == (len(user\_ids), len(item\_ids))

print('Matrix:', sparse\_matrix.shape, 'nnz:', sparse\_matrix.nnz)

val\_interactions = val\_ev.select(\['user\_id', 'track\_id']).rename({'track\_id': 'item\_id'}).unique()

\# ========== 3.1 АУДИО-сигнал: готовые характеристики Spotify ==========

audio\_by\_track = (ctx.filter(pl.col('track\_id').is\_in(item\_ids))

.group\_by('track\_id').mean())

audio\_map = {r\['track\_id']: \[r\[c] for c in AUDIO\_COLS]

for r in audio\_by\_track.iter\_rows(named=True)}

audio\_matrix = np.zeros((len(item\_ids), len(AUDIO\_COLS)), dtype=np.float32)

for iid, idx in item\_to\_idx.items():

if iid in audio\_map:

audio\_matrix\[idx] = audio\_map\[iid]

\# z-score стандартизация по столбцам

audio\_matrix = (audio\_matrix - audio\_matrix.mean(0)) / (audio\_matrix.std(0) + 1e-9)

print('audio\_matrix:', audio\_matrix.shape)

\# ========== 3.2 ТЕКСТ-сигнал: хэштеги -> TF-IDF -> SVD ==========

from sklearn.feature\_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD

tag\_docs = (train\_ev.filter(pl.col('hashtag').is\_not\_null())

.group\_by('track\_id')

.agg(pl.col('hashtag').str.concat(' ').alias('doc')))

doc\_map = dict(zip(tag\_docs\['track\_id'].to\_list(), tag\_docs\['doc'].to\_list()))

corpus = \[doc\_map.get(iid, '') for iid in item\_ids] # строго в порядке item\_to\_idx

tfidf = TfidfVectorizer(max\_features=5000, min\_df=3)

X\_tfidf = tfidf.fit\_transform(corpus)

text\_matrix = TruncatedSVD(n\_components=64, random\_state=RANDOM\_SEED)\\

.fit\_transform(X\_tfidf).astype(np.float32)

print('text\_matrix:', text\_matrix.shape)

\# ========== 3.3 ГРАФ-сигнал: item2vec (word2vec по сессиям) ==========

from gensim.models import Word2Vec

sentences = (train\_ev.sort('created\_at')

.group\_by('user\_id')

.agg(pl.col('track\_id').cast(pl.Utf8).alias('seq'))

)\['seq'].to\_list()

w2v = Word2Vec(sentences=sentences, vector\_size=64, window=5,

min\_count=1, sg=1, epochs=5, seed=RANDOM\_SEED, workers=4)

graph\_matrix = np.zeros((len(item\_ids), 64), dtype=np.float32)

for iid, idx in item\_to\_idx.items():

key = str(iid)

if key in w2v.wv:

graph\_matrix\[idx] = w2v.wv\[key]

print('graph\_matrix:', graph\_matrix.shape)

\# ========== 3.4 МУЛЬТИМОДАЛЬНЫЙ сигнал: готовый эмбеддинг VK-LSVD ==========

from huggingface\_hub import hf\_hub\_download

SUB = 'up0.001\_ip0.001'

vk\_files = (\[f'subsamples/{SUB}/train/week\_{i:02}.parquet' for i in range(25)]

\+ \[f'subsamples/{SUB}/validation/week\_25.parquet', 'metadata/item\_embeddings.npz'])

for f\_ in vk\_files:

hf\_hub\_download(repo\_id='deepvk/VK-LSVD', repo\_type='dataset', filename=f\_, local\_dir='VK-LSVD')

vk\_train = pl.concat(\[pl.scan\_parquet(f'VK-LSVD/subsamples/{SUB}/train/week\_{i:02}.parquet')

for i in range(25)]).collect(engine='streaming')

vk\_val\_interactions = (pl.read\_parquet(f'VK-LSVD/subsamples/{SUB}/validation/week\_25.parquet')

.select(\['user\_id', 'item\_id']).unique())

emb\_npz = np.load('VK-LSVD/metadata/item\_embeddings.npz')

emb\_pos = {int(t): k for k, t in enumerate(emb\_npz\['item\_id'])}

vk\_items = sorted(set(vk\_train\['item\_id'].unique().to\_list()) & set(emb\_pos.keys()))

vk\_users = sorted(vk\_train\['user\_id'].unique().to\_list())

vk\_user\_to\_idx = {u: i for i, u in enumerate(vk\_users)}

vk\_item\_to\_idx = {t: i for i, t in enumerate(vk\_items)}

\# матрица эмбеддингов (мультимодальный сигнал), 32 старшие компоненты

multi\_matrix = np.stack(\[emb\_npz\['embedding']\[emb\_pos\[t]]\[:32]

for t in vk\_items]).astype(np.float32)

\# веса как в основной части: log-время + бонусы действий, клип, фильтр нулей

vk\_w = (vk\_train.filter(pl.col('item\_id').is\_in(vk\_items))

.with\_columns((1.0 + ((pl.col('timespent') / 60).clip(0, 60) + 1.0).log()

\+ pl.col('like').cast(pl.Int32) \* 3.0

\+ pl.col('share').cast(pl.Int32) \* 2.5

\+ pl.col('bookmark').cast(pl.Int32) \* 2.0

\+ pl.col('click\_on\_author').cast(pl.Int32) \* 1.5

\+ pl.col('open\_comments').cast(pl.Int32) \* 1.0

\+ pl.col('dislike').cast(pl.Int32) \* (-2.0)

).clip(0, 100).alias('weight'))

.filter(pl.col('weight') > 0))

r\_ = vk\_w\['user\_id'].replace(vk\_user\_to\_idx).to\_numpy()

c\_ = vk\_w\['item\_id'].replace(vk\_item\_to\_idx).to\_numpy()

vk\_matrix = coo\_matrix((vk\_w\['weight'].to\_numpy(), (r\_, c\_)),

shape=(len(vk\_users), len(vk\_items))).tocsr()

vk\_matrix.eliminate\_zeros()

print('VK-LSVD matrix:', vk\_matrix.shape, 'nnz:', vk\_matrix.nnz)

print('multi\_matrix (VK-LSVD embedding):', multi\_matrix.shape)

signal\_matrices = {'text': text\_matrix, 'audio': audio\_matrix, 'graph': graph\_matrix}

\# ========== 4.1 Оценка (без изменений из основной части) ==========

def recommend\_top\_k\_from\_scores(scores, seen\_idx, k=10):

scores = scores.copy()

if len(seen\_idx):

scores\[np.array(list(seen\_idx))] = -np.inf

top = np.argpartition(-scores, k)\[:k]

return top\[np.argsort(-scores\[top])]

def evaluate\_score\_fn(score\_fn, val\_interactions, sparse\_matrix, user\_to\_idx, item\_to\_idx,

K=10, sample\_size=500, seed=42, item\_idx\_filter=None):

rnd = random.Random(seed)

vf = val\_interactions.filter(pl.col('user\_id').is\_in(list(user\_to\_idx.keys())) &

pl.col('item\_id').is\_in(list(item\_to\_idx.keys())))

vu = vf\['user\_id'].unique().to\_list()

if len(vu) > sample\_size:

vu = rnd.sample(vu, sample\_size)

recalls, ndcgs = \[], \[]

for uid in vu:

ui = user\_to\_idx\[uid]

true\_idx = {item\_to\_idx\[i] for i in

vf.filter(pl.col('user\_id') == uid)\['item\_id'].to\_list() if i in item\_to\_idx}

if item\_idx\_filter is not None:

true\_idx &= item\_idx\_filter

if not true\_idx:

continue

rec = recommend\_top\_k\_from\_scores(score\_fn(ui), sparse\_matrix\[ui].indices, k=K)

hits = len(set(rec) & true\_idx)

recalls.append(hits / min(K, len(true\_idx)))

dcg = sum(1/np.log2(p+2) for p, r in enumerate(rec) if r in true\_idx)

idcg = sum(1/np.log2(p+2) for p in range(min(len(true\_idx), K)))

ndcgs.append(dcg/idcg if idcg > 0 else 0.0)

return {'Recall\@K': float(np.mean(recalls)) if recalls else 0,

'NDCG\@K': float(np.mean(ndcgs)) if ndcgs else 0,

'Tested users': len(recalls)}

\# ========== 4.2 ALS-baseline ==========

model = AlternatingLeastSquares(factors=64, regularization=0.1,

iterations=20, alpha=1.0, random\_state=RANDOM\_SEED)

model.fit(sparse\_matrix)

def als\_scores(ui):

return model.item\_factors @ model.user\_factors\[ui]

print('ALS:', evaluate\_score\_fn(als\_scores, val\_interactions, sparse\_matrix, user\_to\_idx, item\_to\_idx))

\# ========== 4.3 Контент-модель и гибрид для ЛЮБОГО сигнала и ЛЮБОГО датасета ==========

def make\_content\_scores(content\_matrix, inter\_matrix):

ws = inter\_matrix @ content\_matrix

wt = np.asarray(inter\_matrix.sum(axis=1)).flatten(); wt\[wt == 0] = 1.0

up = ws / wt\[:, None]

cn = np.linalg.norm(content\_matrix, axis=1) + 1e-9

def cs(ui, cm=content\_matrix, up=up, cn=cn):

p = up\[ui]

return (cm @ p) / (cn \* (np.linalg.norm(p) + 1e-9))

return cs

def min\_max(x):

lo, hi = x.min(), x.max()

return (x - lo) / (hi - lo) if hi > lo else np.zeros\_like(x)

def make\_hybrid(cf\_fn, content\_fn, alpha):

def h(ui):

return alpha \* min\_max(cf\_fn(ui)) + (1 - alpha) \* min\_max(content\_fn(ui))

return h

\# ========== 5. Сравнение сигналов (nowplaying-RS + мультимодальный на VK-LSVD) ==========

def run\_signal(name, content\_matrix, inter\_matrix, val\_inter, u2i, i2i, cf\_fn, base\_recall):

cs = make\_content\_scores(content\_matrix, inter\_matrix)

solo = evaluate\_score\_fn(cs, val\_inter, inter\_matrix, u2i, i2i)

best = (0, None)

for alpha in \[0.3, 0.5, 0.7, 0.9]:

m = evaluate\_score\_fn(make\_hybrid(cf\_fn, cs, alpha), val\_inter, inter\_matrix, u2i, i2i)

if m\['Recall\@K'] > best\[0]:

best = (m\['Recall\@K'], alpha)

pop = np.asarray(inter\_matrix.sum(axis=0)).flatten()

tail = set(np.argsort(-pop)\[int(0.2 \* inter\_matrix.shape\[1]):].tolist())

h\_tail = evaluate\_score\_fn(make\_hybrid(cf\_fn, cs, best\[1]), val\_inter, inter\_matrix, u2i, i2i,

item\_idx\_filter=tail)

b\_tail = evaluate\_score\_fn(cf\_fn, val\_inter, inter\_matrix, u2i, i2i, item\_idx\_filter=tail)

gain = (best\[0] / base\_recall - 1) \* 100

gain\_tail = ((h\_tail\['Recall\@K'] / b\_tail\['Recall\@K'] - 1) \* 100

if b\_tail\['Recall\@K'] > 0 else float('nan'))

print(f"{name:<14} {solo\['Recall\@K']:>8.4f} {best\[0]:>8.4f} {best\[1]:>6} "

f"{gain:>+9.1f}% {gain\_tail:>+11.1f}%")

return dict(solo=solo\['Recall\@K'], hybrid=best\[0], alpha=best\[1],

gain=gain, gain\_tail=gain\_tail)

\# baseline nowplaying-RS

base\_np = evaluate\_score\_fn(als\_scores, val\_interactions, sparse\_matrix, user\_to\_idx, item\_to\_idx)

print(f"ALS baseline (nowplaying): Recall={base\_np\['Recall\@K']:.4f}")

print(f"{'Сигнал':<14} {'Solo':>8} {'Hybrid':>8} {'alpha':>6} {'прирост':>10} {'прирост tail':>12}")

results = {}

for name, cm in signal\_matrices.items():

results\[name] = run\_signal(name, cm, sparse\_matrix, val\_interactions,

user\_to\_idx, item\_to\_idx, als\_scores, base\_np\['Recall\@K'])

\# мультимодальный — на VK-LSVD: свой ALS и свой baseline

vk\_als = AlternatingLeastSquares(factors=64, regularization=0.1,

iterations=20, alpha=1.0, random\_state=RANDOM\_SEED)

vk\_als.fit(vk\_matrix)

def vk\_als\_scores(ui):

return vk\_als.item\_factors @ vk\_als.user\_factors\[ui]

base\_vk = evaluate\_score\_fn(vk\_als\_scores, vk\_val\_interactions, vk\_matrix,

vk\_user\_to\_idx, vk\_item\_to\_idx)

print(f"ALS baseline (VK-LSVD): Recall={base\_vk\['Recall\@K']:.4f}")

results\['multimodal(VK)'] = run\_signal('multimod.(VK)', multi\_matrix, vk\_matrix,

vk\_val\_interactions, vk\_user\_to\_idx, vk\_item\_to\_idx,

vk\_als\_scores, base\_vk\['Recall\@K'])

\# Сигналы сравниваем по колонкам "прирост" и "прирост tail": они нормированы

\# на собственный baseline и потому сопоставимы между датасетами.

\# ========== 6. Permutation importance по блокам сигналов ==========

from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.inspection import permutation\_importance

from sklearn.model\_selection import train\_test\_split

N\_COMP, N\_SAMPLE = 8, 200\_000

samp = train\_w\.sample(n=min(N\_SAMPLE, len(train\_w)), seed=RANDOM\_SEED)

iidx = samp\['track\_id'].replace(item\_to\_idx).to\_numpy()

blocks, X\_parts, feat\_names = {}, \[], \[]

for name, cm in signal\_matrices.items():

n = min(N\_COMP, cm.shape\[1])

X\_parts.append(cm\[iidx, :n])

cols = \[f'{name}\_{i}' for i in range(n)]

blocks\[name] = cols; feat\_names += cols

X = np.hstack(X\_parts).astype(np.float32)

y = np.log1p(samp\['weight'].to\_numpy())

Xtr, Xte, ytr, yte = train\_test\_split(X, y, test\_size=0.2, random\_state=RANDOM\_SEED)

gbm = HistGradientBoostingRegressor(max\_iter=150, random\_state=RANDOM\_SEED).fit(Xtr, ytr)

print(f'R^2: {gbm.score(Xte, yte):.3f}')

pi = permutation\_importance(gbm, Xte\[:20000], yte\[:20000], n\_repeats=3, random\_state=RANDOM\_SEED)

imp = dict(zip(feat\_names, pi.importances\_mean))

print('\n--- Важность по блокам ---')

for b, cols\_ in blocks.items():

print(f'{b:<12} {sum(imp\[c] for c in cols\_):.4f}')
