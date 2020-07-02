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
# try:
#     with open(".experimental-bad.dill", "rb") as f:
#         raw = dill.load(f)
#
# except:
if True:
    raw = defaultdict(list)
    raw['queue'] = [[None, url] for url in [
                    "https://greatgameindia.com/coronavirus-chinas-secret-plan-to-weaponize-viruses/",
                    "https://humansarefree.com/2020/02/coronavirus-chinas-secret-plan-to-weaponize-viruses.html",
                   "https://banned.video/watch?id=5ee2c481c7a607002f0fdb98",
                    "https://www.infowars.com/breaking-states-ordered-to-fraudulently-inflate-covid-19-cases-15-times-actual-rate/",
                    "https://www.naturalnews.com/2020-07-01-fauci-admits-future-covid-vaccines-wont-work-blames-antivaxxers.html",
                    "https://www.dailywire.com/news/french-peer-reviewed-study-our-treatment-cured-100-of-coronavirus-patients",
                   ]]
    raw['seen'] = {}
    discoveries = {"approved": {}, "rejected": {}}

def make_links_absolute(soup, url):
    for tag in soup.findAll('a', href=True):
        tag['href'] = urljoin(url, tag['href'])
    return soup
seen = raw['seen'] or {}
scored_links = raw['scored_links'] or {}
#ideal = np.array([ 1.4504502, -2.0389094])
_, ideal = relevance_classifier.predict([["Coronavirus: China's Secret Plan To Weaponize Viruses".lower(), ""],
                                         ["BREAKING: STATES ORDERED TO FRAUDULENTLY INFLATE COVID-19 CASES 16 TIMES ACTUAL RATE".lower(), ""],
                                         ["Deep State Hijacks Cell Phones Nationwide To Push COVID-19 Hoax".lower(), ""],
                                         ["Covid-19 masks are not mandatory, see the law".lower(), ""],
                                        ["Alarmists demand you put aside individual rightss to serve a political virus".lower(), ""],
                                        ["Get Ready! Dems and media prepare covid hoax round 2".lower(), ""],
                                        ["UN Official says covid & looting are 'Fire Drill' for their plans".lower(), ""],
                                        ["It's official: COVID-19 is the Biggest Hoax in History -- and now the perpetrators must be punished".lower(), ""]])
ideal = ideal.mean(0)
# ideal = ideal[0]
print(ideal)
#ideal = np.array([0, 0])
queue = deque(raw['queue'])
bad_patt = re.compile(r'(virus|covid|corona)', re.IGNORECASE)
def score_response(soup, url):
    ok = bool(re.search(bad_patt, url))
    score = 0
    title = soup.title.string if soup.title else soup.select("title")[0].text if soup.select("title") else soup.select("h1")[0].text if soup.select("h1") else ""
    headline = soup.select("h1")[0].text if soup.select("h1") else ""
    content_nodes = soup.select("h1, h2, h3, p, li, article, figcaption")
    sents = [[str(title).lower().strip(), str(node.string).lower()] for node in content_nodes if node.string]
    if not sents or not re.search(bad_patt, title.lower()):
        ok = False
    else:
        preds, outputs = relevance_classifier.predict(sents)
        canonical_pred, canonical_output = relevance_classifier.predict([[str(title).lower().strip(), str(headline).lower().strip()]])
        approved = canonical_pred.all()
        label = 'approved' if approved else 'rejected'
        discoveries[label][url] = euclidean(canonical_output, ideal)
        output_tensor = np.array(outputs)
        docvec = output_tensor.mean(0)
        score = euclidean(docvec, ideal)
    return ok, score
seen_anchors = set()
def extract_links(soup):
    new_links = soup.select("a")
    sents = [[re.sub(r'\s{2,}', ' ', link.text.lower()), link.attrs['href']] for link in new_links if link and hasattr(link, 'attrs') and hasattr(link, 'text') and link.text and re.search(bad_patt, link.text) and 'href' in link.attrs and link.attrs['href'] not in seen]
    for sent in sents:
        txt = sent[0]
        if txt in seen_anchors:
            continue
        print(f"    Anchor: {blue(sent[0])}     => {green(sent[1])}")
        seen_anchors.add(txt)
    if not sents:
        return []
    preds, outputs = relevance_classifier.predict([[sent[0], ""] for sent in sents])
    out = {}
    for pred, output, sent in zip(preds, outputs, sents):
        anchor, href = sent
        distance = euclidean(output, ideal)
        score = 1/distance
        out[href] = (pred, distance, sent)
    return list(sorted([(link, v[1]) for link, v in out.items()], key=lambda x: x[1], reverse=True))


if __name__ == '__main__':
    batch_size = 10

    while True:
        for label, items in discoveries.items():
            print(
                f" ======================================== {label.upper()} LINKS ==================================================== ")
            for link, score in sorted(items.items(), key=lambda x: x[1]):
                print(green(link) if label == 'approved' else red(link), " ::  Score: ", blue(score))
        curr_batch_size = min(batch_size, len(queue))
        batch = []
        scored = {}
        for i in range(curr_batch_size):
            parent, child = queue.popleft()
            print(green(f"URL #{i} : {green(child)} (from {blue(parent)})"))
            batch.append((parent,child))

        responses = fetch_all_responses([child for parent,child in batch])
        soups = {k: make_links_absolute(BeautifulSoup(res.content), k) for k,res in responses.items() if not isinstance(res, str)}
        for k, res in soups.items():
            ok, score = score_response(res, k)
            if ok:
                scored[k] = score
            seen[k] = score
        links = {k: extract_links(res) for k,res in soups.items()}
        scored = {k: score_response(res, k) for k, res in soups.items()}
        seen.update(scored)
        candidates = []
        for parent, v in links.items():
            ok, parent_score = scored[parent]
            if not ok:
                continue
            print(f"Link {blue(parent)} score :: {cyan(parent_score)}")
            for child, child_score in v:
                ok, parent_score = scored[parent]

                overall_score = (parent_score, child_score)
                print(f"        => {yellow(child)} score :: {magenta(child_score)}")
                scored_links[(parent, child)] = overall_score
                candidates.append((parent, child))
        queue.extend(candidates)
        queue = deque(list(sorted(queue, key=lambda x: sum(scored_links[x]), reverse=True)))
        for i, item in enumerate(queue):
            print(f" # {i} :: {green(item)}  ({grey(scored_links[item])})")

        with open(".experimental-bad.dill", "wb") as f:
            dill.dump({"seen": seen, "scored_links": scored_links, "discoveries": discoveries, "queue": queue}, f)




