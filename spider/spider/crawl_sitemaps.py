import random
import datetime
import json
from spider.async_utils import fetch_all_responses
from spider.common import config, init_db, init_conn, create_sitemaps_table, create_spiderqueue_table, create_articles_table
MAX_REQUESTS = 100

conn = init_conn()
db = init_db()
create_sitemaps_table()
create_spiderqueue_table()
create_articles_table()

def main():
    responsedb = db['sitemaps']
    sitemaps = {datetime.datetime.now().strftime(row['url']): row for row in db['sitemapindex']}
    for row in sitemaps.values():
        print(json.dumps(row, indent=4, default=str))
    urls = list( sitemaps.keys())
    urls = random.sample(urls, len(urls))
    results = fetch_all_responses(urls, MAX_REQUESTS)
    responses = []
    for url, result in results.items():
        row = sitemaps[url]
        row['url'] = url
        if result:
            try:
                row['resolved_url'] = url
                row['content'] = result.content
                row['bytes'] = len(result.content)
                row['status_code'] = int(result.status_code)
                row['status_msg'] = result.reason_phrase
                row['ok'] = int(result.status_code) == 200
                row['response_headers'] = dict(result.headers)
                row['encoding'] = result.encoding
                row['created_at'] = datetime.datetime.now()
                row['is_sitemap'] = True
                row['is_content'] = False
            except Exception as e:
                row['resolved_url'] = url
                row['ok'] = False
                row['error'] = result
                row['created_at'] = datetime.datetime.now()
                row['encoding'] = 'utf-8'
                row['status_code'] = 0
                row['status_msg'] = f"{e.__class__.__name__} :: Response timed out. Error message: {e}"
                row['is_sitemap'] = True
                row['is_content'] = False
                row['bytes'] = 0
        else:
            row['created_at'] = datetime.datetime.now()
            row['error'] = result

        responses.append(row)
        print(row)

    responsedb.upsert_many(responses, ['url'], ensure=True)

if __name__ == '__main__':
    main()




