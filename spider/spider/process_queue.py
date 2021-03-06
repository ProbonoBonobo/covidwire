from spider.common import init_db
from spider.async_utils import fetch_all_responses
from spider.utils import extract_content
from gemeinsprache.utils import green, cyan, yellow, red, magenta, blue, grey
from date_guesser import guess_date
from dateutil.parser import parse as parse_timestamp
from unicodedata import normalize
from lxml.html import fromstring as parse_html
import re
from simpletransformers.classification import ClassificationModel
import os
import json
from ftfy import fix_text, fix_text_segment
from bs4 import BeautifulSoup
import datetime
from spider.utils import DotDict, Haystack
import torch
from scipy.spatial.distance import cosine
from metadata_parser import MetadataParser
from flatdict import FlatDict

audience_labels = [
    "international",
    "city",
    "regional",
    "national",
    "indefinite",
    "state",
]
LIMIT = os.getenv("SPIDER_LIMIT", 5000)
MAX_REQUESTS = os.getenv("MAX_REQUESTS", 50)



NULL_DATE = datetime.datetime(1960, 1, 1)
relevance_classifier = ClassificationModel(
    "roberta",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "lib/classification_model/v1.0",
    ),
    use_cuda=0*int(torch.cuda.is_available()),
)
audience_classifier = ClassificationModel(
    "roberta",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "lib/audience_classifier/v1.0",
    ),
    num_labels=len(audience_labels),
    use_cuda=0 * int(torch.cuda.is_available())
)

UniversalSelector = DotDict(
    {
        "content": "div[class*='-content'] > p, div[class*='content-'] > p, section > p, div[class*='body'] > p, div[class*='article'] > p, div[itemprop*='article'] > p, div[class*='text'] > p, section div[class*='inner'] > p, #article-content p, div[class*='article-body'] p, div[class*='premium'] p, div[class*='story'] p, article.story-body p, div.body-text, article > p, article p, .postBody p, div[class*='main-content'] p, .inner p, #content p, .postBody",
        "title": "//meta[@name='twitter:title' or @property='og:title' or @name='title']/@content",
        "published_at": "//meta[contains(@property, 'article:published_time') or contains(@name, 'parsely-pub-date') or contains(@property, 'datePublished') or contains(@property, 'parsely-pub-date') or @property='tncms-access-version' or contains(@property,'pubDate') or contains(@itemprop, 'date') or contains(@name, 'ublished') or contains(@property, 'article_date') or contains(@property, 'sailthru.date') or contains(@property, 'article:publishedtoweb') or @name='date' or @name='last_updated_date']/@content | //time/text() | //*[contains(@class, 'time') or contains(@id, 'time') or contains(@aria-label, 'a.m') or contains(@aria-label, 'p.m.') or contains(@aria-label, '2020')]/text() | //time/@datetime | //time/text() | //*[@id='story-info']/small/text()",
        "summary": "//meta[@property='og:description' or @name='twitter:description' or contains(@property,'description') or contains(@name,'description')]/@content",
        "author": "//meta[contains(@property, 'author') or contains(@name, 'author') or @property='byline' or @property='article_author' or @property='og:sitename' or @name='twitter:site']/@content",
        "image_url": "//meta[@property='og:image' or @name='twitter:image' or contains(@property, 'thumbnail') or contains(@name, 'thumbnail')]/@content",
        "section": "//meta[@property='og:section' or @property='parsely-section' or contains(@name, 'section') or @property='article:section' or contains(@property, 'section')]/@content",
        "keywords": "//meta[@property='article:tag' or @name='parsely-tags' or contains(@name, 'keyword') or contains(@name, 'tag') or contains(@property, 'keyword') or contains(@name, 'tag')]/@content",
        "name": "//meta[@property='og:site_name' or @name='application-name']/@content"
    }
)


class Article:
    odors = re.compile(
        r"(please|sorry,|subscr|comment|social network|\||share|facebook|twitter|{|function\s?\(|click|advert|reached at|rights reserved|top stories|©|affiliate|Associated Press|writer|email|permission|curated by|delivered every|news stor|privacy|copyright|link)",
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
    passthrough_attrs = ("url", "city", "state", "loc", "site", "name", "is_dumpsterfire", "selector")

    def __init__(self, url, html, row, soup=None, lxml=None, fix_encoding_errors=True):
        self.url = url
        self.sitemap_data = row
        self.html = (
            fix_text_segment(html.replace("\xa0", " "), uncurl_quotes=False)
            if fix_encoding_errors
            else html
        )
        try:

            self.soup = soup if soup else BeautifulSoup(self.html)
        except Exception as e:
            raise ValueError(f"{e.__class__.__name__} :: {e}, {self.html}")
        self.meta = Haystack(html)
        #print(json.dumps(self.meta, indent=4))
        try:
            if isinstance(self.html, str):
                self.html = self.html.encode("utf-8")
            self.lxml = lxml if lxml else parse_html(self.html)
        except Exception as e:
            raise ValueError(f"{e.__class__.__name__} :: {e}, {self.html}")
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
            "metadata": {k: v for k, v in self.meta.data.items()},
        }
        self.data.update(
            {k: row[k] for k in self.passthrough_attrs if row and k in row}
        )

    @staticmethod
    def is_xpath(s):
        return s and isinstance(s, str) and s.startswith("/")

    @property
    def published_at(self):
        date = NULL_DATE

        def dt_specificity(dt):
            return sum(1 if date.__getattribute__(k) else 0 for k in ('month', 'day', 'hour', 'minute', 'second', 'microsecond', 'tzinfo'))

        if "lastmod" in self.sitemap_data and self.sitemap_data["lastmod"] and (datetime.datetime.now() - self.sitemap_data['lastmod']) < datetime.timedelta(days=365*10):
            date = self.sitemap_data["lastmod"]
        else:
            # print(f"Guessing date for {self.url}...")
            # guess = guess_date(url=self.url, html=self.html)
            # date = guess.date or NULL_DATE
            # if date is not NULL_DATE:
            #     print(f"Found a date: {guess.date}! Accuracy is to: {guess.accuracy}")
            # else:
            #     print(
            #         f"No date extracted. Attempting to find a date in the metadata..."
            #     )
                most_specific = None
                curr_specificity = 0
                seen = set()
                for match in self.lxml.xpath(UniversalSelector.published_at):
                    if match in seen:
                        continue
                    seen.add(match)
                    try:
                        date = parse_timestamp(str(re.sub(r"\n", " ", match.strip())))
                        print(f"Found a date! {date}")
                        specificity = dt_specificity(date)
                        if specificity > curr_specificity :
                            most_specific = date
                            curr_specificity = specificity
                    except:
                        pass
                if most_specific:
                    date = most_specific
        try:
            date = date.replace(tzinfo=None)
        except:
            pass

        return date

    @property
    def keywords(self):
        for match in self.lxml.xpath(UniversalSelector.keywords):
            result = str(match).strip()
            if result:
                return [m.strip() for m in re.split(r"\s*,\s*", result)]
        return self.meta.re_search([r"(Article:keywords:\d+)"])

    @property
    def content(self):
        content = ""
        if 'selector' in row and row['selector']:
            content = extract_content(self.soup, row['selector'])
            if len(content) >= 250:
                return content
            nodes = self.soup.select(UniversalSelector.content)
        else:
            nodes = self.soup.select(UniversalSelector.content)
        txt = []
        if not list(nodes):
            nodes = self.soup.select("p")
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
        for match in self.lxml.xpath(UniversalSelector.section):
            result = ', '.join(set([s.strip() for s in re.findall(r"[\w\s/&+]+", str(match).strip())]))
            if result:
                return result

        targets = ["NewsArticle:articleSection", "Article:articleSection"]
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
            needle = re.sub(r"\\u.+$", "", re.sub(r"^.+http", "http", needle))
        for candidate in self.lxml.xpath(UniversalSelector.image_url):
            if isinstance(candidate, str):
                url = candidate.strip()
                return url
            return result


print(f"Initializing db...")
db = init_db()
print("initialized.")
import random
if __name__ == "__main__":
    url2tensor = {}

    spiderqueue = db["spiderqueue"]
    table = db["articles_v2"]
    dumpsterfire = db["dumpsterfire"]
    candidate_articles = []
    quarantined_articles = []

    passthrough_attrs = ("url", "city", "state", "loc", "site", "name", "selector")
    print(f"Loading seen urls...")
    good_urls = db.query("SELECT url FROM articles_v2 a;")
    bad_urls = db.query("select url from dumpsterfire b;")
    seen = set()
    seen.update([row["url"] for row in good_urls])
    seen.update([row["url"] for row in bad_urls])
    seen.update([row['url'] for row in db.query("select url from articles a;")])
    print(f"Loaded {len(list(seen))} seen urls.")
    rows = {
        row["url"]: row
        for row in db.query(f"select * from spiderqueue where lastmod is not null order by lastmod desc limit 5000;")
        if row["url"] not in seen
    }
    print(f"Found {len(list(rows))} uncrawled urls...")
    urls = list(rows.keys())[0 : min(LIMIT, len(list(rows.keys())))]
    random.shuffle(urls)
    # urls = random.sample(urls, k=len(urls))

    print(
        green(f"[ process_queue ] :: Added {len(list(rows.keys()))} urls to the queue.")
    )
    responses = fetch_all_responses(urls, MAX_REQUESTS)

    for url, res in responses.items():
        row = rows[url]
        row["prediction"] = "rejected"
        row["mod_status"] = "rejected"
        is_dumpsterfire = row["is_dumpsterfire"]

        if isinstance(res, str):
            row["ok"] = False

            if is_dumpsterfire:
                dumpsterfire.upsert(row, ["url"])
                print(f"Inserted bad response from {row['url']} to dumpsterfire")
                print(red(f"Response text: {res}"))
            else:
                table.upsert(row, ["url"])
                print(f"Inserted bad response from {row['url']} to {table.name}")
                print(red(f"Response text: {res}"))

            continue
        html = res.decoded
        try:
            article = Article(url, html, row)
        except ValueError as e:
            row["ok"] = False
            row["error_msg"] = str(e)

            print(red(e))

            if is_dumpsterfire:
                dumpsterfire.upsert(row, ["url"])
                print(f"Inserted {row['url']} to dumpsterfire")
            else:
                table.upsert(row, ["url"])
                print(f"Inserted {row['url']} to {table.name}")
            continue
        row = article.data
        # print(json.dumps(row, indent=4, default=lambda x: str(x) if isinstance(x, datetime.datetime) else x))
        if not all(k in row and row[k] for k in ("content", "title")):
            missing_attrs = [
                k for k in ("content", "title") if k not in row or not row[k]
            ]
            row["ok"] = False
            row[
                "error_msg"
            ] = f"ParseError :: Failed to parse {len(missing_attrs)} required attributes : {missing_attrs}"
            print(
                red(
                    json.dumps(
                        row,
                        indent=4,
                        default=lambda x: str(x)
                        if isinstance(x, datetime.datetime)
                        else x,
                    )
                )
            )
            if is_dumpsterfire:
                dumpsterfire.upsert(row, ["url"])
                print(f"Inserted bad row {row['url']} to dumpsterfire")
            else:
                table.upsert(row, ["url"])
                print(f"Inserted bad row {row['url']} to {table.name}")

            continue

        try:
            tokens = list([t.strip() for t in re.split(r"\s+", row["content"].lower())])
            truncated_preview = " ".join(tokens[: min(100, len(tokens))])
            model_input = [row["title"].lower(), truncated_preview]
            prediction, relevance_tensor = relevance_classifier.predict([model_input])
            row["prediction"] = ["rejected", "approved"][prediction[0]]
            prediction, audience_tensor = audience_classifier.predict([row["title"]])
            vec = relevance_tensor[0].tolist() + audience_tensor[0].tolist()
            row["docvec_v2"] = vec
            row["docvec_version"] = "classifier_outputs"

            print(
                f"Audience prediction: {prediction} ({audience_labels[prediction[0]]})"
            )
            print(
                f"Relevance classifier output (shape: {relevance_tensor.shape}): {blue(relevance_tensor)}"
            )
            print(
                f"Audience classifier output (shape: {audience_tensor.shape} : {blue(audience_tensor)}"
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
            if is_dumpsterfire:
                dumpsterfire.upsert(row, ["url"])
                print(f"Inserted {row['url']} to dumpsterfire")
            else:
                table.upsert(row, ["url"])
                print(f"Inserted {row['url']} to {table.name}")
        except Exception as e:
            print(red(f"ERROR: {e.__class__.__name__} :: {e}", row))
            continue
    # candidates = list(url2tensor.values())
    # rel_tensors = [(url, candidate[0]) for url, candidate in url2tensor.items()]
    # audience_tensors = [(url, candidate[1]) for url, candidate in url2tensor.items()]
    # for url1, tensor1 in url2tensor.items():
    #     rel_tensor1, audience_tensor1 = tensor1
    #     k_nearest_rel = dict(
    #         list(sorted(rel_tensors, key=lambda x: cosine(x[1], rel_tensor1)))[0:5]
    #     )
    #     k_nearest_aud = dict(
    #         list(
    #             sorted(audience_tensors, key=lambda x: cosine(x[1], audience_tensor1))
    #         )[0:5]
    #     )
    #     print(
    #         f"================================= URL: {url1} ============================================="
    #     )
    #     print(magenta("Relevance k neighbors :"))
    #     for i, rest in enumerate(k_nearest_rel.items()):
    #         url2, tensor2 = rest
    #         print(f"    {i}.  {green(url2)} ({cosine(rel_tensor1, tensor2)})")
    #         print(f"           {grey(rel_tensor1)}")
    #         print(f"           {grey(tensor2)}")
    #     print("                              * * * * * * * * * * * * * * * * * * * * *")
    #     print(blue("Audience k neighbors :"))
    #     for i, rest in enumerate(k_nearest_aud.items()):
    #         url2, tensor2 = rest
    #         print(f"    {i}.  {green(url2)} ({cosine(audience_tensor1, tensor2)})")
    #         print(f"           {grey(audience_tensor1)}")
    #         print(f"           {grey(tensor2)}")
