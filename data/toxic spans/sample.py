# %%
import numpy as np
import pandas as pd
from tqdm import tqdm

MIN_SIZE = 50

# %% load data:
comments    = pd.read_csv('comments.csv', index_col=0)
annotations = pd.read_csv('annotations.csv', index_col=0)
spans       = pd.read_csv('spans.csv')

# %% fix types:
spans.dropna(inplace=True)
spans.loc[spans.type == 'Insu', 'type'] = 'Insult'
spans.loc[spans.type == 'Identity based ', 'type'] = 'Identity based Attack'

# %% compute spans:
for a, c in tqdm(zip(annotations.index, annotations['comment_id'].values), total=len(annotations)):
    for _, s in spans[spans['annotation'] == a].iterrows():
        tp = s.type.lower()

        if tp not in comments.columns:
            comments[tp] = [[] for _ in range(len(comments))]
        
        try: comments.loc[c, tp].append((s.start, s.end))
        except KeyError: continue

# %% delete rows with multiple classes:
labels = ['insult', 'identity based attack', 'profane/obscene', 'threat', 'other toxicity']
labels = [
    [l for s, l in zip(row, labels) if len(s) > 0]
    for row in tqdm(comments[labels].values)
]
mask = np.array([len(l) == 1 for l in labels])

sample = comments[mask]
sample['label'] = [l[0] for l, m in zip(labels,mask) if m]
sample['spans']  = [list(set(sample.loc[i,l])) for i,l in zip(sample.index, sample.label.values)]

# %% filter duplicate texts:
sample['comment_text'] = [t for t in sample['comment_text']]
sample['comment_normed'] = [t.lower() for t in sample['comment_text']]
sample = sample.loc[sample['comment_normed'].drop_duplicates().index]

# %%  only texts longer than MIN_SIZE:
sample['comment_size'] = [len(t) for t in sample['comment_text']]
sample = sample[sample['comment_size'] >= MIN_SIZE]

# %%  only texts with only one annotated span:
sample['spans_size'] = [len(t) for t in sample['spans']]
sample = sample[sample['spans_size'] == 1]

# %% extract spans:
sample['spans'] = [
    txt[spn[0][0]:spn[0][1]]
    for txt, spn in sample[['comment_text', 'spans']].values
]

# %% delete unneeded columns:
sample = sample[['comment_text', 'label', 'spans']]

# %% take sample of 200 comments:
comment_ids = []
for tp in ['insult', 'identity based attack', 'profane/obscene', 'threat']:
    comment_ids.extend(np.random.choice(sample[sample.label == tp].index, 50, False))
sample = sample.loc[comment_ids].sort_index()

# %% save sample:
sample.to_csv('sample.csv')