import pandas as pd

MIN_SIZE = 50

# load incidents:
incidents = pd.read_csv('food_recall_incidents.csv', index_col=0).fillna('')
incidents['key'] = [f'[{str(y)}-{str(m)}-{str(d)}] "{str(t)}"' for y, m, d, t in incidents[['year', 'month', 'day', 'title']].values]

# filter columns:
incidents = incidents[[
        'year', 'month', 'day',
        'title',
#        'product-raw', 'product-category',
        'hazard-raw', 'hazard-category',
        'language', 'country'
]]

# only english  samples:
incidents = incidents[incidents['language'] == 'en']

# filter duplicate texts:
incidents['title-normed'] = [t.lower().strip() for t in incidents.title]
incidents = incidents.loc[incidents['title-normed'].drop_duplicates().index]

# only texts longer than MIN_SIZE:
incidents['title-size'] = [len(t) for t in incidents.title]
incidents = incidents[incidents['title-size'] >= MIN_SIZE]

# drop "other hazard":
incidents = incidents[incidents['hazard-category'] != 'other hazard']


# only samples with hazard-category in text:
incidents = incidents[[hazard in title for title, hazard in incidents[['title', 'hazard-raw']].values]]

# take random sample of 200 texts:
incidents = incidents[[
    'year', 'month', 'day',
    'title',
    'hazard-category',
    'hazard-raw',
    'language', 'country'
]].sample(200).reset_index(drop=True)

# save sample:
incidents.to_csv('incidents_sample.csv')