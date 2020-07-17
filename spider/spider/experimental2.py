from unidecode import unidecode
import numpy as np
from sentence_transformers import SentenceTransformer
import re
import dataset
import os
from collections import OrderedDict
from gemeinsprache.utils import *
import dill
from collections import deque
import trio
from urllib.parse import urljoin, urlparse
from simpletransformers.classification import ClassificationModel
import numpy as np
import random
from gemeinsprache.utils import *
import re
from spider.async_utils import fetch_all_responses
from ftfy import fix_text_segment
import os
from collections import defaultdict
from url_normalize import url_normalize
from bs4 import BeautifulSoup
from scipy.spatial.distance import euclidean, cosine
import dill
NLP_MODEL_ID = 'roberta-large-nli-mean-tokens'
model = SentenceTransformer(NLP_MODEL_ID)

example_headlines = """Time to Escape America is Running Out as EU considers BANNING Americans from Entering the EU over Coronavirus
Coronavirus Numbers Purposefully Inflated By The CDC
SHOCK! Amazon and eBay remove all Coronavirus products from Sale
Pakistani Muslim DELIBERATELY Spread Coronavirus in Italy, Infected Hundreds of People
Chinese caught INTENTIONALLY spreading Coronavirus in USA
China Scams World Into Believing they 'Healed' Everyone of Coronavirus by Turning on Empty Factories
Worldwide Population Being Tortured In Deep State Psyop
COVID-19 has Exposed Ugliness of GLOBALISM & OPEN BORDERS and given Nations incentive to regain INDEPENDENCE
Governor Abbott Confronted With Evidence COVID-19 Is A Giant Hoax
Covid-19 is deadly, but it will never kill the relentless stupidity of Leftists
US Government officially Labels COVID-19 a Chinese Artificial Bioweapon Escaped from Wuhan Lab
Breaking: States Ordered To Fraudulently Inflate COVID-19 Cases 15 Times Actual Rate
Collective Torture For A Virus That Has Never Been Shown To Exist
Romania develops Coronavirus Vaccine able to Cure White People only
EXPOSED! Disgusting Chinese Sparked Coronavirus Outbreak by Eating BATS ON VIDEO!
British man shocks doctors after he totally CURES CORONAVIRUS with 'hot whisky and honey'
China Apocalypse: People dropping dead everywhere, cremated in secret, its a total Disaster
Globalists WANT Coronavirus to spread everywhere and kick-start the Apocalypse
Chinese scientists: Coronavirus is a man-made virus from a Laboratory in Wuhan
Accelerationist Pope Francis Trying to spread COVID19 urges priests to 'visit, comfort coronavirus sufferers'
Bill Gates Wants to Vaccinate the Entire Planet for Coronahoax with a 666 Digital ID
Chinese Scumbags SCAM Spain of $500 Million Amid Coronavirus, Send them Faulty Tests
California Hospitals Threaten to FIRE Nurses for Wearing N95 Masks while Treating Coronavirus Patients
Stanford study: COVID-19 much more widespread than thought, and NO MORE DEADLY THAN FLU
BREAKING: US Withdraws From World Health Organization and Democrats Are Losing It
Bus driver beaten by passengers refusing to wear face masks left brain-dead
Just be done with it and nuke the bastards - Two-year-old girl 'is raped while in Covid-19 isolation ward' at South African hospital
Coronavirus Vaccines, Depopulation and the Demonic War to Claim your Soul for Satan""".split("\n")

start_urls = ["https://banned.video/",
              "https://www.prisonplanet.com/",
              "https://rense.com/",
              "https://outragedepot.com/",
              "https://www.rt.com/",
              "https://www.eutimes.net/",
              "https://www.naturalnews.com/",
              "https://www.godlikeproductions.com/?c1=1&c2=1&disclaimer=Continue",
              "https://www.foxnews.com/",
              "http://www.atlanteanconspiracy.com/",
              "https://www.gopusa.com/",
              "http://cuttingthroughthematrix.com/",
              "http://www.rumormillnews.com/",
              "https://davidicke.com/category/coronavirus/",
              "http://www.abovetopsecret.com/",
              "http://www.thetruthseeker.co.uk/",
              "http://www.whatreallyhappened.com/",
              "https://www.rt.com/search?q=coronavirus&type=Post",
              "https://www.rt.com/search?q=coronavirus&type=News&xcategory=sport",
              "https://www.infowars.com/",
              "https://www.brighteon.com/categories/4ad59df9-25ce-424d-8ac4-4f92d58322b9",
              "https://www.breitbart.com/tag/coronavirus/",
              "https://www.theblaze.com/",
              "https://www.drudgereport.com/",
              "https://www.wnd.com/?s=coronavirus",
              "https://www.foxnews.com/category/health/infectious-disease/coronavirus",
              "https://www.youtube.com/watch?v=ZyNG0TInJN8",
              "https://voat.co/"]

def make_links_absolute(soup, url):
    for tag in soup.findAll('a', href=True):
        tag['href'] = urljoin(url, tag['href'])
    return soup

def extract_links(soup):
    nodes = soup.select("a")
    url2anchor = defaultdict(set)
    for link in nodes:
        try:
            anchor = fix_text_segment(re.sub(r"\s+", " ", link.text.strip()))
            url = url_normalize(link.attrs['href'].strip())
            parsed = urlparse(url)

        except:
            continue
        if url not in visited and anchor:
            # if anchor and len(anchor) > 10:
            url2anchor[url].add(anchor)
            if not seen[url]:
                alt_anchor = ' '.join([w for w in re.findall(r"([\w/]+?/?$)", parsed.path) if w.isalpha()]).title()
                if len(alt_anchor) > 20:
                    url2anchor[url].add(alt_anchor)
        # seen[url].add(anchor)
    return url2anchor


    # sents = [[re.sub(r'\s{2,}', ' ', link.text.lower()), link.attrs['href']] for link in new_links if link and hasattr(link, 'attrs') and hasattr(link, 'text') and link.text and 'href' in link.attrs and link.attrs['href'] not in seen]
    # for sent in sents:
    #     txt = sent[0]
    #     if txt in seen_anchors:
    #         continue
    #     print(f"    Anchor: {blue(sent[0])}     => {green(sent[1])}")
    #     seen_anchors.add(txt)
    # if not sents:
    #     return []
    # preds, outputs = relevance_classifier.predict([[sent[0], ""] for sent in sents])
    # out = {}
    # for pred, output, sent in zip(preds, outputs, sents):
    #     anchor, href = sent
    #     distance = euclidean(output, ideal)
    #     score = 1/distance
    #     out[href] = (pred, distance, sent)
    # return list(sorted([(link, v[1]) for link, v in out.items()], key=lambda x: x[1], reverse=True))


vecs = model.encode([fix_text_segment(headline.strip()) for headline in example_headlines])
tensor = np.array(vecs)
centroid = tensor.mean(0)
print(centroid)
batch_size = 100
seen = defaultdict(set)
vecs = {}
visited = set()
queue = start_urls
curr_iter = 0
while True:
    curr_batch_size = min(batch_size, len(queue))
    batch = queue[:curr_batch_size]
    queue = queue[curr_batch_size:]
    responses = fetch_all_responses(batch, 200)
    visited.update(batch)
    soups = {k: make_links_absolute(BeautifulSoup(res.content), k) for k, res in responses.items() if
             not isinstance(res, str)}
    changed = set()
    for page, soup in soups.items():
        url2anchors = extract_links(soup)
        for url, anchors in url2anchors.items():
            seen_anchors = seen[url]
            if any(anchor not in seen_anchors for anchor in anchors):
                changed.add(url)
                seen[url].update(anchors)
    sample = set(random.sample([url for url in seen if any(re.search(r"(virus|covid|mask|hoax|conspiracy|fauci|lockdown|pandemic)", anchor, re.IGNORECASE) for anchor in seen)], k=5))
    print(cyan(f"Sample of the extracted anchor texts:"))
    for k in sample:
        print(blue(json.dumps({k: seen[k]}, indent=4, default=list)))
    vecs.update({url: np.array(model.encode(list(seen[url]))).mean(0) for url in changed})
    scored = defaultdict(lambda: 999999, {url: cosine(vec, centroid) for url, vec in vecs.items()})
    queue = list(sorted(list(set(seen.keys()).difference(visited)), key=lambda url: scored[url]))
    for i, url in enumerate(queue[0:20]):
        score = scored[url]
        anchors = seen[url]
        print(green(f"    QUEUE #{i} :: {url}"), grey(f"(Cosine distance: {score})"))
        for anchor in anchors:
            print(cyan(f"            >>  {anchor}"))
    print(grey(" . . . "))
    for i, url in enumerate(queue[-5:]):
        score = scored[url]
        anchors = seen[url]
        print(grey(f"    QUEUE #{len(queue)-5+i} :: {url}"), grey(f"(Cosine distance: {score})"))
        for anchor in anchors:
            print(grey(f"            >>  {anchor}"))
    if not curr_iter % 5:
        global_ranks = list(sorted([url for url in seen], key=lambda x: scored[x]))
        data = {url: {"rank": i, "anchors": list(seen[url]), "score": scored[url]} for i, url in enumerate(global_ranks) if url in vecs and url in seen}
        with open("experimental3.json", "w") as f:
            json.dump(data, f, indent=4)
    curr_iter += 1


    #links = dict(**extract_links(res) for k, res in soups.items())
    #scored = {k: score_response(res, k) for k, res in soups.items()}
    #seen.update(scored)
