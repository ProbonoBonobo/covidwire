import json
import re
import dataset
import psycopg2
import os
from collections import OrderedDict

target = "prod"
config = {
    "local": {
        "user": "kz",
        "password": "admin",
        "host": "127.0.0.1",
        "port": "5432",
        "database": "cvwire",
    },
    "staging": {
        "user": "postgres",
        "password": "admin",
        "host": "34.83.188.109",
        "port": "5432",
        "database": "postgres",
    },
    "prod": {
        "user": "admin",
        "password": "Feigenbum4",
        "host": "64.225.121.255",
        "port": "5432",
        "database": "covidwire",
    },
}

from collections import defaultdict, deque
def init_db(target=target):
    db_config = config[target]
    return dataset.connect(
        f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    )


db = init_db()
cols = (
    "docvec_v2",
    "title",
    "name",
    "description",
    "content",
    "published_at",
    "image_url",
    "prediction",
    "audience",
    "url",
)
if True:
    corpus = []
    table = db['geolabeled_articles']
    articles_v1 = db['articles']
    articles_v2 = db['articles_v2']

    index = defaultdict(list)
    seen = set()
    dups = set()
    for row in table:
        #index[row['label']].append(row)
        if row['title'] in seen:
            dups.add(row['title'])
        seen.add(row['title'])
    for row in table:
        if row['title'] in dups:
            continue
        index[row['label']].append(row)
    filtered = {k:v for k,v in index.items() if len(v) >= 15}
    for k,v in filtered.items():
        for row in v:
            url = row['url']
            result = articles_v2.find_one(url=url)
            if not result:
                result = articles_v1.find_one(url=url)
            if not result:
                print(f"Still no results for url {url}: {row['title']}")

                continue
            if not all(result[k] for k in ('content', 'description', 'title')):
                continue
            title = ' '.join([w.strip() for w in re.split(r"\s+", result['title'].strip())])
            description = ' '.join([w.strip() for w in re.split(r"\s+", result['description'].strip())])
            preview = re.split(r"\s+", result['content'].strip())
            preview = ' '.join([w.strip() for w in preview[:min(len(preview), 200)]]) + "..."
            keywords = result['keywords']
            if keywords:
                keywords = [[k, k[1:-1]][bool(k.startswith("\""))] for k in keywords[1:-1].split(",")]
                keywords = "{" + ', '.join(keywords[:min(len(keywords), 10)]) + "}"

            elif not keywords:
                keywords = "{}"

            stub = f"Headline: {result['title']} \nSource: {result['name']} \nKeywords: {keywords} \nDescription: {result['description']} \nPreview: {preview}"
            corpus.append((stub, k))
            print(stub)
    all_labels = list(sorted(list(filtered.keys())))
    docs = []
    targets = []
    import random
    for doc, target in random.sample(corpus, len(corpus)):
        docs.append(doc)
        _target = all_labels.index(target)
        targets.append(_target)
    dataset = {"data": docs, "targets": targets, "labels": all_labels}
    with open("/home/kz/datasets/geotagged-articles-training-set_v6.json", "w") as f:
        json.dump(dataset, f)


