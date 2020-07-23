import pickle
import datetime
import json
from spider.common import config, init_db
from gemeinsprache.utils import cyan, red, blue, green, yellow, magenta
from dateutil.parser import parse as parse_timestamp
import pytz
from bs4 import BeautifulSoup

MAX_REQUESTS = 10
MAX_ARTICLES_PER_SOURCE = 250
NULL_DATE = datetime.datetime(1960, 1, 1)

db = init_db()

seen = set()


def ensure_tztime(ts):
    if isinstance(ts, str):
        ts = parse_timestamp(ts)
    try:
        return pytz.utc.localize(ts)
    except:
        return ts


def parse_sitemap(row, seen):
    if not row or not row["content"]:
        return [], set()
    sitemap_url = row["url"]

    soup = BeautifulSoup(row["content"], "xml")
    elements = soup.findAll("url")
    rows = []
    for elem in elements:
        url_node = elem.find("loc")
        lastmod_node = elem.find("lastmod")
        xmlmeta = "\n".join([str(e) for e in elem.children]).encode("utf-8")

        try:
            lastmod = parse_timestamp(lastmod_node.text.strip())
        except:
            lastmod = NULL_DATE

        try:
            url = url_node.text.strip()
        except:
            continue

        if url:
            url = url.strip()
            if url in seen:
                continue

            row = {
                "url": url.strip(),
                "site": row["site"],
                "name": row["name"],
                "city": row["city"],
                "state": row["state"],
                "loc": row["loc"],
                "lastmod": lastmod,
                "xmlmeta": xmlmeta,
                "is_dumpsterfire": row["is_dumpsterfire"],
                "selector": row['selector']
            }
            if row["url"] not in seen:
                rows.append(row)
                print(
                    blue(
                        json.dumps(
                            row,
                            indent=4,
                            default=lambda x: str(x)
                            if isinstance(x, (bytes, datetime.datetime))
                            else x,
                        )
                    )
                )
                seen.add(url.strip())
            if len(rows) > MAX_ARTICLES_PER_SOURCE:
                break

    rows = list(
        sorted(rows, key=lambda row: ensure_tztime(row["lastmod"]), reverse=True)
    )
    print(
        magenta("[ fetch_sitemap ] "),
        f":: Extracted {len(rows)} urls from sitemap: {sitemap_url}",
    )
    return rows, seen


if __name__ == "__main__":
    crawldb = db["articles"]
    responsedb = db["sitemaps"]
    spiderqueue = db["spiderqueue"]
    dumpsterfire = db["dumpsterfire"]
    seen.update([row["url"] for row in db.query("select url from spiderqueue")])
    seen.update([row["url"] for row in db.query("select url from articles")])
    seen.update([row["url"] for row in db.query("select url from dumpsterfire")])
    # seen.update([row['url'] for row in dumpsterfire])

    queue = [row for row in responsedb]

    parsed = []
    for row in queue:
        rows, _seen = parse_sitemap(row, seen)
        seen.update(_seen)
        parsed.extend(rows)
        if rows:
            print(f"Inserting {len(rows)} rows...")
            spiderqueue.upsert_many(rows, ["url"])
            print(f"Insert complete.")
    # print(f"Upserting {len(parsed)} rows...")
    # spiderqueue.insert_many(parsed, ['url'])
    print(f"Update complete.")
