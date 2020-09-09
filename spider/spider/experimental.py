from pymagnitude import Magnitude
from spider.async_utils import fetch_all_responses
import re
import numpy as np
from lxml.html import fromstring as parse_html
from collections import defaultdict
from scipy.spatial.distance import cosine
from gemeinsprache.utils import *
model =  Magnitude("/home/kz/datasets/glove-lemmatized.6B.50d.magnitude")

samples = [('Imperial County Copes With COVID Increase',
  'Imperial County, as with other California counties, has seen an increase in positive tests for the coronavirus and cases of COVID-19. How is this mostly rural agricultural region coping with the pandemic?'),
           ("Reps4Vets program aids Veterans, combats suicide at 4:13 Gym",
            "IMPERIAL &mdash; Amid the freshly recycled air and gym goers working out, scattered about due to social-distancing, three soldiers began their leg day workouts with a spark of hope."),
           ("Imperial County receives acceptance of variance, moves forward in reopening",
            "EL CENTRO — Imperial County Public Health Director Janette Angulo announced the California Department of Public Health and Governor Newsom’s approval of Imperial County’s variance report during a Roadmap to Recovery update Tuesday, August 25."),
 ("What's Behind A COVID-19 Spike In Imperial County",
  'The rural county on the U.S.-Mexico border with 180,000 residents has the highest coronavirus infection rate per capita of any California county.'),
 ("'It's Too Little, Too Late': Sentiments From El Centro Residents Amid Growing COVID-19 Cases",
  'The small border towns of El Centro and Calexico in Imperial County are getting national attention due to their growing number of coronavirus cases. Imperial...')]
scored = defaultdict(lambda: 1)
queue = ["https://www.kqed.org/news/11824749/whats-behind-a-covid-19-spike-in-imperial-county",
        "https://www.thedesertreview.com/news/local/"]
title_tensor = []
desc_tensor = []
audience_labels = [
    "international",
    "city",
    "regional",
    "national",
    "indefinite",
    "state",
]
from simpletransformers.classification import ClassificationModel

audience_classifier = ClassificationModel(
    "roberta",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "lib/audience_classifier/v1.0",
    ),
    num_labels=len(audience_labels),
    use_cuda=True,

)
def doc2vec(s, s2=None):
    entry = [s, s2] if s2 else [s]
    predictions, raw_outputs = audience_classifier.predict(entry)
    if s2:
        raw_outputs = raw_outputs.mean(0)
    return raw_outputs

for title, desc in samples:
    title_tensor.append(doc2vec(title))
    desc_tensor.append(doc2vec(desc))

t1 = np.array(title_tensor).mean(0)
t2 = np.array(desc_tensor).mean(0)
T = np.array([t1, t2]).mean(0)
results = []
seen = set()
while queue:
    partition = min(len(queue), 5)
    batch, queue = queue[:partition], queue[partition:]
    print(f"Next batch:")
    for url in batch:
        print(green(url), blue(scored[url]))
    for parent_url, res in fetch_all_responses(batch).items():
        print(f"Parsing {parent_url}...")

        if isinstance(res, str):
            continue


        dom = parse_html(res.content)
        dom.make_links_absolute(parent_url)
        title = dom.xpath("//meta[@name='twitter:title' or @property='og:title' or @name='title']/@content")
        description = dom.xpath("//meta[@property='og:description' or @name='twitter:description' or contains(@property,'description') or contains(@name,'description')]/@content")
        for link in dom.xpath("//a"):
            url = link.xpath("./@href")
            text = link.text_content()
            if url and text and url[0] not in seen:
                scored[url[0]] = cosine(doc2vec(text), T)
                queue.append(url[0])
                seen.add(url[0])

        if not title or not description:
            continue
        v1 = doc2vec(title[0], description[0])
        V = v1
        score = cosine(V, T)
        # results.append((title[0], description[0], score))
        print(f"url {parent_url} has score {score}")
        scored[title[0]] = score
        results.append((title[0], parent_url))
    queue = list(sorted(list(set(queue)), key=lambda x: scored[x], reverse=False))
    results = list(sorted(results, key=lambda x: scored[x[0]], reverse=False))

    print(f"Top 20 matches so far:")
    seen = set()
    i = 0
    for title, url in results:
        if title in seen:
            continue
        i += 1
        seen.add(title)
        score = scored[title]

        print(f"{i} :: {green(title)} (Distance: {score})")
        if i >= 50:
            break


