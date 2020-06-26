from spider.common import init_db, init_conn
from spider.async_utils import fetch_all_responses
from gemeinsprache.utils import green, blue, cyan, yellow, red, magenta
from psycopg2.extras import RealDictCursor
import time
import typing
import os
from collections import defaultdict

import json
import subprocess
from date_guesser import guess_date
from enum import Enum
from dateutil.parser import parse as parse_timestamp
from unicodedata import normalize
from lxml.html import fromstring as parse_html
import re
from simpletransformers.classification import ClassificationModel
import os
import json
from ftfy import fix_text
import random
from bs4 import BeautifulSoup, element, UnicodeDammit
import datetime
from spider.utils import DotDict, Haystack

audience_labels = [
    "international",
    "city",
    "regional",
    "national",
    "indefinite",
    "state",
]
LIMIT = 500
MAX_REQUESTS = 10


NULL_DATE = datetime.datetime(1960, 1, 1)
relevance_classifier = ClassificationModel(
    "roberta",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "lib/classification_model/v1.0",
    ),
    use_cuda=0,
)
audience_classifier = ClassificationModel(
    "roberta",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "lib/audience_classifier/v1.0",
    ),
    num_labels=len(audience_labels),
    use_cuda=0,
)

UniversalSelector = DotDict(
    {
        "content": "div[class*='-content'] > p, div[class*='content-'] > p, section > p, div[class*='body'] > p, div[class*='article'] > p, div[itemprop*='article'] > p, div[class*='text'] > p, section div[class*='inner'] > p, #article-content p, div[class*='article-body'] p, div[class*='premium'] p, div[class*='story'] p, article.story-body p, div.body-text, article > p, article p, .postBody p, div[class*='main-content'] p, .inner p, #content p, .postBody",
        "title": "//meta[@name='twitter:title' or @property='og:title' or @name='title']/@content",
        "published_at": "//meta[contains(@property, 'article:published_time') or contains(@property, 'datePublished') or contains(@property, 'parsely-pub-date') or @property='tncms-access-version' or contains(@property,'pubDate') or contains(@name, 'ublished') or contains(@property, 'article_date') or contains(@property, 'sailthru.date') or contains(@property, 'article:publishedtoweb') or @name='date' or @name='last_updated_date']/@content | //time/text() | //*[contains(@class, 'time') or contains(@id, 'time') or contains(@aria-label, 'a.m') or contains(@aria-label, 'p.m.') or contains(@aria-label, '2020')]/text()",
        "summary": "//meta[@property='og:description' or @name='twitter:description' or contains(@property,'description') or contains(@name,'description')]/@content",
        "author": "//meta[contains(@property, 'author') or contains(@name, 'author') or @property='byline' or @property='article_author' or @property='og:sitename' or @name='twitter:site']/@content",
        "image_url": "//meta[@property='og:image' or @name='twitter:image' or contains(@property, 'thumbnail') or contains(@name, 'thumbnail')]/@content",
    }
)


class Article:
    odors = re.compile(
        r"(please|subscr|comment|social network|\||share|facebook|twitter|{|function\s?\(|click|advert|reached at|rights reserved|top stories|©|affiliate|Associated Press|writer|email|permission|curated by|delivered every|news stor|privacy|copyright|link)",
        re.IGNORECASE,
    )
    func_attrs = (
        "content",
        "url",
        "title",
        "published_at",
        "summary",
        "author",
        "image_url",
    )
    passthrough_attrs = ("url", "city", "state", "loc", "site", "name")

    def __init__(self, url, html, row, soup=None, lxml=None, fix_encoding_errors=True):
        self.url = url
        self.sitemap_data = row
        self.html = fix_text(html.replace("\xa0", " ")) if fix_encoding_errors else html
        self.soup = soup if soup else BeautifulSoup(self.html)
        self.meta = Haystack(self.soup)
        print(json.dumps(self.meta, indent=4))
        self.lxml = lxml if lxml else parse_html(self.html)
        self.data = {
            "content": self.content,
            "url": self.url,
            "title": self.title,
            "published_at": self.published_at,
            "description": self.summary,
            "author": self.author,
            "image_url": self.image_url,
            "section": self.section,
            "publisher": self.publisher,
            "keywords": self.keywords,
            "metadata": {k: v for k, v in self.meta.items()},
        }
        self.data.update(
            {k: row[k] for k in self.passthrough_attrs if row and k in row}
        )

    @staticmethod
    def is_xpath(s):
        return s and isinstance(s, str) and s.startswith("/")

    @property
    def published_at(self):
        if "lastmod" in self.sitemap_data and self.sitemap_data["lastmod"]:
            date = self.sitemap_data["lastmod"]
        else:
            print(f"Guessing date for {self.url}...")
            guess = guess_date(url=self.url, html=self.html)
            date = guess.date or NULL_DATE
            if date is not NULL_DATE:
                print(f"Found a date: {guess.date}! Accuracy is to: {guess.accuracy}")
            else:
                print(
                    f"No date extracted. Attempting to find a date in the metadata..."
                )
                for match in self.lxml.xpath(UniversalSelector.published_at):
                    try:
                        date = parse_timestamp(str(match))
                        print(f"Found a date! {date}")
                        break
                    except:
                        pass
        try:
            date = date.replace(tzinfo=None)
        except:
            pass

        return date

    @property
    def keywords(self):
        return self.meta.re_search([r"(Article:keywords:\d+)"])

    @property
    def content(self):
        txt = []
        nodes = self.soup.select(UniversalSelector.content)
        for node in nodes:
            if (
                node
                and node.string
                and node.string.strip()
                and not re.search(self.odors, node.string)
            ):
                txt.append(
                    normalize(
                        "NFKD", fix_text(node.string.replace("\xa0", " "))
                    ).strip()
                )
        body = "\n".join(txt)
        return body

    @property
    def title(self):
        for candidate in self.lxml.xpath(UniversalSelector.title):
            candidate = str(candidate)
            longest = list(sorted(re.split(r"\s[|-]\s", candidate), key=len))[-1]
            formatted = normalize("NFKD", fix_text(longest.strip()))
            return formatted

    @property
    def summary(self):
        meta_targets = ["NewsArticle:description", "Article:description"]
        needle = self.meta.search(meta_targets)
        if needle:
            return needle
        for candidate in self.lxml.xpath(UniversalSelector.summary):
            txt = normalize(
                "NFKD", fix_text(str(candidate).replace("\xa0", " "))
            ).strip()
            html_removed = BeautifulSoup(txt, "lxml").text.strip()
            return html_removed

    @property
    def publisher(self):
        targets = ["NewsArticle:publisher:name"]
        needle = self.meta.search(targets)
        return needle

    @property
    def section(self):
        targets = ["NewsArticle:articleSection"]
        needle = self.meta.search(targets)
        return needle

    @property
    def author(self):
        targets = ["NewsArticle:author:name", "NewsArticle:author:0:name"]
        for candidate in self.lxml.xpath(UniversalSelector.author):
            txt = normalize(
                "NFC", fix_text(str(candidate).replace("\xa0", " "))
            ).strip()
            return txt

    @property
    def image_url(self):
        result = None
        meta_targets = [
            "NewsArticle:image:url",
            "NewsArticle:image",
            "NewsArticle:thumbnailUrl",
            "Article:image",
            "Article:image:url",
            "Article:thumbnailUrl",
            "OpinionNewsArticle:image",
            "ReportageNewsArticle:image",
            "NewsArticle:image:0:url",
            "Article:image:0:url",
            "ImageObject:url",
            "Organization:logo",
            "NewsArticle:publisher:logo:url",
            "Article:publisher:logo:url",
        ]
        needle = self.meta.search(meta_targets)
        if needle:
            return needle
        for candidate in self.lxml.xpath(UniversalSelector.image_url):
            if isinstance(candidate, str):
                url = candidate.strip()
                return url
        return result


if __name__ == "__main__":
    conn = init_conn()
    db = init_db()
    table = db["articles"]

    passthrough_attrs = ("url", "city", "state", "loc", "site", "name")
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(f"""
            SELECT *
            FROM   spiderqueue q
            WHERE  NOT EXISTS (
                SELECT  -- SELECT list mostly irrelevant; can just be empty in Postgres
                FROM   articles a
                WHERE  a.url = q.url
         ) ORDER BY lastmod desc limit {LIMIT};
    """)
        rows = {row["url"]: row for row in cur.fetchall()}
    print(green("[ process_queue ] :: Added {len(rows)} urls to the queue."))
    urls = list(rows.keys())[0 : min(LIMIT, len(list(rows.keys())))]
    urls = random.sample(urls, k=len(urls))
    responses = fetch_all_responses(urls, MAX_REQUESTS)
    articles = []
    for url, res in responses.items():
        row = rows[url]
        if isinstance(res, str):
            continue
        html = res.decoded
        article = Article(url, html, row)
        row = article.data
        # print(json.dumps(row, indent=4, default=lambda x: str(x) if isinstance(x, datetime.datetime) else x))
        if not all(k in row and row[k] for k in ("content", "title")):
            row["ok"] = False
            articles.append(row)
            continue

        try:
            tokens = list([t.strip() for t in re.split(r"\s+", row["content"].lower())])
            truncated_preview = " ".join(tokens[: min(100, len(tokens))])
            model_input = [row["title"].lower(), truncated_preview]
            prediction, out = relevance_classifier.predict([model_input])
            row["prediction"] = ["rejected", "approved"][prediction[0]]
            prediction, out = audience_classifier.predict([row["title"]])
            print(
                f"Audience prediction: {prediction} ({audience_labels[prediction[0]]})"
            )
            row["audience"] = audience_labels[prediction[0]]
            row["mod_status"] = (
                "pending" if row["prediction"] == "approved" else "rejected"
            )
            try:
                print(
                    magenta(
                        json.dumps(
                            row,
                            indent=4,
                            default=lambda x: x
                            if not isinstance(x, (datetime.datetime, dict))
                            else str(x),
                        )
                    )
                )
            except Exception as e:
                print(e.__class__.__name__, e)
            print(
                "\n\n======================================================================================\n"
            )
            print(f"Article: {green(row['title'])} ({yellow(row['name'])})")
            print(cyan(truncated_preview) + "\n")
            print(
                f"PREDICTION: {red(row['prediction'].upper()) if row['prediction'] == 'rejected' else green(row['prediction'].upper())}"
            )
            articles.append(row)
        except Exception as e:
            print(e.__class__.__name__, e)

        if len(articles) > 50:
            print(cyan(f"[ process_queue ] :: Inserting {len(articles)} articles..."))
            try:
                table.upsert_many(articles, ["url"])
                articles = []
                for article in articles:
                    print(green(f"[ process_queue ] :: Inserted {article['url']}"))
            except Exception as e:
                print(f"no insert : {e.__class__.__name__} :: {e}")
                continue
    if articles:
        print(cyan(f"[ process_queue ] :: Inserting {len(articles)} articles..."))
        table.upsert_many(articles, ["url"])
        for article in articles:
            print(green(f"[ process_queue ] :: Inserted {article['url']}"))
