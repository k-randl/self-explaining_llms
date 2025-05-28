# %%
import numpy as np
import pandas as pd

# %% load data:
comments    = pd.read_csv('comments.csv', index_col=0)
annotations = pd.read_csv('annotations.csv', index_col=0)
spans       = pd.read_csv('spans.csv')

# %% fix types:
spans.dropna(inplace=True)
spans.loc[spans.type == 'Insu', 'type'] = 'Insult'
spans.loc[spans.type == 'Identity based ', 'type'] = 'Identity based Attack'

# %% take sample of 200 comments:
comment_ids = np.random.choice(annotations.comment_id.unique(), 200, False)
sample = comments.loc[comment_ids] 

# %% compute spans:
for a, c in zip(annotations.index, annotations['comment_id'].values):
    for _, s in spans[spans['annotation'] == a].iterrows():
        if s.type not in sample.columns:
            sample[s.type] = [[] for _ in comment_ids]
        
        try: sample.loc[c, s.type].append((s.start, s.end))
        except KeyError: continue

# %% save sample:
sample.to_csv('sample.csv')