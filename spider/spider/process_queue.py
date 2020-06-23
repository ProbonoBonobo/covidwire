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
from bs4 import BeautifulSoup, element, UnicodeDammit
import datetime
from spider.utils import DotDict

audience_labels = ['international', 'city', 'regional', 'national', 'indefinite', 'state']
LIMIT = 1000
MAX_REQUESTS = 100

conn = init_conn()
db = init_db()
NULL_DATE = datetime.datetime(1960, 1, 1)
relevance_classifier = ClassificationModel(
    "roberta", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib/classification_model/v1.0"), use_cuda=0
)
audience_classifier = ClassificationModel(
    "roberta", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib/audience_classifier/v1.0"), num_labels=len(audience_labels), use_cuda=0
)


UniversalSelector = DotDict(
    {
        "content": "div[class*='-content'] > p, div[class*='content-'] > p, section > p, div[class*='body'] > p, div[class*='article'] > p, div[itemprop*='article'] > p, div[class*='text'] > p, section div[class*='inner'] > p, #article-content p, div[class*='article-body'] p, div[class*='premium'] p, div[class*='story'] p, article.story-body p, div.body-text, article > p, article p, .postBody p, div[class*='main-content'] p, .inner p, #content p, .postBody",
        "title": "//meta[@name='twitter:title' or @property='og:title' or @name='title']/@content",
        "published_at": "//meta[contains(@property, 'article:published_time') or contains(@property, 'datePublished') or contains(@property, 'parsely-pub-date') or @property='tncms-access-version' or contains(@property,'pubDate') or contains(@name, 'ublished') or contains(@property, 'article_date') or contains(@property, 'sailthru.date') or contains(@property, 'article:publishedtoweb') or @name='date' or @name='last_updated_date']/@content | //time/text() | //*[contains(@class, 'time') or contains(@id, 'time') or contains(@aria-label, 'a.m') or contains(@aria-label, 'p.m.') or contains(@aria-label, '2020')]/text()",
        "summary": "//meta[@property='og:description' or @property='twitter:description' or contains(@name,'description')]/@content",
        "author": "//meta[contains(@property, 'author') or contains(@name, 'author') or @property='byline' or @property='article_author' or @property='og:sitename' or @property='twitter:site']/@content",
    }
)


class Article:
    odors = re.compile(
        r"(please|subscr|comment|social network|\||share|facebook|twitter|{|function\s?\(|click|advert|reached at|rights reserved|top stories|©|affiliate|Associated Press|writer|email|permission|curated by|delivered every|news stor|privacy|copyright|link)",
        re.IGNORECASE,
    )

    def __init__(self, url, html, soup=None, lxml=None, fix_encoding_errors=True):
        self.url = url
        self.html = fix_text(html.replace("\xa0", " ")) if fix_encoding_errors else html
        self.soup = soup if soup else BeautifulSoup(self.html)
        self.lxml = lxml if lxml else parse_html(self.html)
        self.data = {
            "content": self.content,
            "url": self.url,
            "title": self.title,
            "published_at": self.published_at,
            "summary": self.summary,
            "author": self.author,
        }

    @staticmethod
    def is_xpath(s):
        return s and isinstance(s, str) and s.startswith("/")

    @property
    def published_at(self):
        print(f"Guessing date for {self.url}...")
        guess = guess_date(url=self.url, html=self.html)
        date = guess.date or NULL_DATE
        if date is not NULL_DATE:
            print(f"Found a date: {guess.date}! Accuracy is to: {guess.accuracy}")
        else:
            print(f"No date extracted. Attempting to find a date in the metadata...")
            for match in self.lxml.xpath(UniversalSelector.published_at):
                try:
                    date = parse_timestamp(str(match))
                    print(f"Found a date! {date}")
                    break
                except:
                    continue
        try:
            date = date.replace(tzinfo=None)
        except:
            pass
        return date

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
        for candidate in self.lxml.xpath(UniversalSelector.summary):
            txt = normalize(
                "NFKD", fix_text(str(candidate).replace("\xa0", " "))
            ).strip()
            html_removed = BeautifulSoup(txt, "lxml").text.strip()
            return html_removed

    @property
    def author(self):
        for candidate in self.lxml.xpath(UniversalSelector.author):
            txt = normalize(
                "NFKD", fix_text(str(candidate).replace("\xa0", " "))
            ).strip()
            return txt


table = db["articles"]

seen = set([row["url"] for row in table if row["mod_status"] is None])
if True:
    passthrough_attrs = ("url", "city", "state", "loc", "site", "name")

    tmpdir = f"/tmp/cvwire{int(time.time())}"
    tmpfiles = {}
    os.makedirs(tmpdir, exist_ok=True)
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            f"select  * from spiderqueue where random() < 0.3 order by lastmod desc limit {LIMIT};"
        )
        rows = {row["url"]: row for row in cur.fetchall() if row["url"] not in seen}
    urls = list(rows.keys())
    responses = fetch_all_responses(urls)
    articles = []
    for url, res in responses.items():
        row = rows[url]
        if isinstance(res, str):
            continue
        article = Article(url, res.text)
        row = dict(article.data, **{k: row[k] for k in passthrough_attrs})
        if row["published_at"] == NULL_DATE and "lastmod" in row and row["lastmod"]:
            row["published_at"] = row["lastmod"]
            row["modified_at"] = row["lastmod"]
        else:
            row["modified_at"] = row["published_at"]
        try:
            tokens = list([t.strip() for t in re.split(r"\s+", row["content"].lower())])
            truncated_preview = " ".join(tokens[: min(100, len(tokens))])
            model_input = [row["title"].lower(), truncated_preview]
            prediction, out = relevance_classifier.predict([model_input])
            row["prediction"] = ["rejected", "approved"][prediction[0]]
            prediction, out =  audience_classifier.predict([row['title']])
            print(f"Audience prediction: {prediction} ({audience_labels[prediction[0]]})")
            row['audience'] = audience_labels[prediction[0]]
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
        try:
            table.upsert(row, ["url"])
            print(f"inserted {url}")
        except Exception as e:
            print(f"no insert {url} : {e.__class__.__name__} :: {e}")
            continue
