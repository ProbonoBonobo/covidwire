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


def init_db(target=target):
    db_config = config[target]
    return dataset.connect(
        f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    )

db = init_db()
cols = ("docvec_v2", "title", "name", "description", "content", "published_at", "image_url", "prediction", "audience")
import numpy as np
from gemeinsprache.utils import green, red, yellow, blue, cyan, magenta, grey
from textwrap import wrap
def main():
    table = db['articles']
    publication_names = ['The Atlantic', 'The New Yorker', 'The Economist', 'ProPublica', 'Pro Publica', 'Columbia Journalism Review', 'New York Magazine', 'New York Times', 'Vanity Fair', "Harper's Bazaar"]#[row['name'] for row in db.query("select distinct (name) from articles;")]
    articles_per_source = 100
    by_source = {k: [] for k in publication_names}
    for publication in publication_names:
        print(f"Fetching results for publication: {green(publication)}")
        articles = [{k:v for k,v in row.items() if k in cols} for row in table.find(name=publication) if row['docvec_v2'] and len(row['docvec_v2']) == 8]
        if len(articles) < 200:
            continue
        # sort articles by perplexity, which is roughly the inverse of the approval classifier's 2D output vector magnitude
        print(f"{cyan(len(articles))} articles found. Sorting by classification perplexity...")
        ordered_articles = list(sorted(articles, key=lambda row: np.linalg.norm(row['docvec_v2'][:2])))
        print("\n\n", "=" * 10, f"Top 10 Results", "=" * 10)

        for i, article in enumerate(ordered_articles[0:min(len(ordered_articles), 10)]):
            vec = article['docvec_v2'][:2]
            x, y = vec
            approved = x < y
            title = article['title'] or ""
            desc = article['description'] or ""
            mag = np.linalg.norm(article['docvec_v2'][:2])
            color = green if approved else red
            print(grey(f"{i} ::"), f"{color(title)} {grey('(' + str(round(mag, 2)) + ')')}")
            for line in wrap(desc):
                print(grey(f"     {line}"))
            print("")

        print("\n\n", "=" * 10, f"Bottom 10 Results", "=" * 10)

        for i, article in enumerate(ordered_articles[min(10, len(ordered_articles)) * -1:]):
            vec = article['docvec_v2'][:2]
            x, y = vec
            approved = x < y
            title = article['title'] or ""
            desc = article['description'] or ""
            mag = np.linalg.norm(article['docvec_v2'][:2])
            color = green if approved else red
            print(grey(f"{i}."), f"{color(title)} {grey('(' + str(round(mag, 2)) + ')')}")
            for line in wrap(desc):
                print(grey(f"     {line}"))
            print("")

        import math
        print(f"Generating perplexity-biased random sample...")
        exponential_dist = np.random.exponential(0.5, size=len(ordered_articles))
        probability_vec = np.array(list(sorted(exponential_dist/exponential_dist.sum(), reverse=True)))
        sampled = np.random.choice(articles, p=probability_vec, size=min(len(ordered_articles), articles_per_source*5), replace=False)
        print("\n\n", cyan("=" * 20), magenta(f"Sampled articles for {publication}:"), cyan("=" * 20))
        pos = []
        neg = []
        results = []
        i = 0
        for article in sorted(sampled, key=lambda x: np.linalg.norm(x['docvec_v2'][:2])):
            vec = article['docvec_v2'][:2]
            x, y = vec
            approved = x < y
            title = article['title'] or ""
            desc = article['description'] or ""
            mag = np.linalg.norm(article['docvec_v2'][:2])
            color = green if approved else red

            target = pos if approved else neg
            if len(target) < math.ceil(articles_per_source / 2):
                i += 1
                target.append((mag, title))
                print(grey(f"{i} ::"), f"{color(title)} {grey('(' + str(round(mag, 2)) + ')')}")

                for line in wrap(desc):
                    print(grey(f"     {line}"))
                print("")
                article['classification_perplexity'] = mag
                db['training_queue_v2'].upsert(article, ['url'])
                print(f"Inserted {color(article)}")
                if i >= articles_per_source:
                    break

        by_source[publication] = results
    return by_source

if __name__ == '__main__':
    main()






