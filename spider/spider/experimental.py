from collections import deque
import trio
from simpletransformers.classification import ClassificationModel
import numpy as np
import random
from gemeinsprache.utils import *
import re
from spider.async_utils import fetch_all_responses
import os
from collections import defaultdict
from bs4 import BeautifulSoup
from scipy.spatial.distance import euclidean, cosine
import dill
relevance_classifier = ClassificationModel(
    "roberta",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "lib/classification_model/v1.0",
    ),
    use_cuda=1,
)
from urllib.parse import urlparse, urljoin
with open(".experimental.dill", "rb") as f:
    raw = dill.load(f)
approved_discoveries = raw['approved_discoveries']
print(f" ======================================== DISCOVERED LINKS ==================================================== ")
for link, score in sorted(approved_discoveries.items(), key=lambda x: x[1]):
    print(green(link), " ::  Score: ", blue(score))
def make_links_absolute(soup, url):
    for tag in soup.findAll('a', href=True):
        tag['href'] = urljoin(url, tag['href'])
    return soup
seen = raw['seen']
scored_links = raw['scored_links']
ideal = np.array([ 1.4504502, -2.0389094])
queue = deque(raw['queue'])

def score_response(soup, url):

    title = soup.title.string if soup.title else soup.select("title")[0].text if soup.select("title") else soup.select("h1")[0].text if soup.select("h1") else ""
    headline = soup.select("h1")[0].text if soup.select("h1") else ""
    content_nodes = soup.select("h1, h2, h3, p, li, article, figcaption")
    sents = [[str(title).lower().strip(), str(node.string).lower()] for node in content_nodes if node.string]
    if not sents:
        return 0
    preds, outputs = relevance_classifier.predict(sents)
    canonical_pred, canonical_output = relevance_classifier.predict([[str(title).lower().strip(), str(headline).lower().strip()]])
    if canonical_pred.all():
        approved_discoveries[url] = euclidean(canonical_output, ideal)
    output_tensor = np.array(outputs)
    docvec = output_tensor.mean(0)
    distance = euclidean(docvec, ideal)
    return distance

def extract_links(soup):
    new_links = soup.select("a")
    sents = [[re.sub(r'\s{2,}', ' ', link.text.lower()), link.attrs['href'].lower()] for link in new_links if link and hasattr(link, 'attrs') and 'href' in link.attrs and link.attrs['href'] not in seen]
    for sent in sents:
        print(f"    Anchor: {blue(sent[0])}     => {green(sent[1])}")
    if not sents:
        return []
    preds, outputs = relevance_classifier.predict(sents)
    out = {}
    for pred, output, sent in zip(preds, outputs, sents):
        anchor, href = sent
        distance = euclidean(output, ideal)
        score = 1/distance
        out[href] = (pred, distance, sent)
    return list(sorted([(link, v[1]) for link, v in out.items()], key=lambda x: x[1], reverse=True))


if __name__ == '__main__':
    batch_size = 50
    while True:
        curr_batch_size = min(batch_size, len(queue))
        batch = []
        for i in range(curr_batch_size):
            parent, child = queue.popleft()
            print(green(f"URL #{i} : {green(child)} (from {blue(parent)})"))
            batch.append((parent,child))

        responses = fetch_all_responses([child for parent,child in batch])
        soups = {k: make_links_absolute(BeautifulSoup(res.content), k) for k,res in responses.items() if not isinstance(res, str)}
        scored = {k: score_response(res, k) for k, res in soups.items()}
        links = {k: extract_links(res) for k,res in soups.items()}
        seen.update(scored)
        candidates = []
        for parent, v in links.items():
            parent_score = scored[parent]
            print(f"Link {blue(parent)} score :: {cyan(parent_score)}")
            for child, child_score in v:
                overall_score = (scored[parent], child_score)
                print(f"        => {yellow(child)} score :: {magenta(child_score)}")
                scored_links[(parent, child)] = overall_score
                candidates.append((parent, child))
        queue.extend(candidates)
        queue = deque(list(sorted(queue, key=lambda x: sum(scored_links[x]), reverse=True)))
        for i, item in enumerate(queue):
            print(f" # {i} :: {green(item)}  ({grey(scored_links[item])})")

        with open(".experimental.dill", "wb") as f:
            dill.dump({"seen": seen, "scored_links": scored_links, "approved_discoveries": approved_discoveries, "queue": queue}, f)




