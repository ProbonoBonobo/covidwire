from ner.common import init_db
import spacy
from allennlp.predictors import Predictor
import allennlp
import allennlp_models
import textwrap
import datetime

from collections import defaultdict
import re
import json
from gemeinsprache.utils import red, green, cyan, magenta, blue, yellow
import torch

LIMIT_ARTICLES = 9999999
TABLE = "articles_v2"
db = init_db()
crawldb = db[TABLE]
allennlp_model = None

import allennlp_models.tagging


def extract_entities_with_allennlp(*s):
    model_url = "https://storage.googleapis.com/allennlp-public-models/ner-model-2020.02.10.tar.gz"
    global allennlp_model
    if not "allennlp_model" in globals() or not globals()["allennlp_model"]:
        print(yellow("[ model_init ]"), f" :: Loading AllenNLP NER model...")

        if torch.cuda.is_available():
            cuda_device = 0
        else:
            cuda_device = -1
        allennlp_model = Predictor.from_path(model_url, cuda_device=cuda_device)
        print(
            yellow(f"[ model_init ] "),
            f" :: CUDA initialized? ",
            [green("YES"), red("NO")][abs(cuda_device)],
        )
        print(yellow(f"[ model_init ] "), f" :: Load complete.")
    print(yellow(f"[ model_predict ]"), f" :: Extracting entities...")
    start = datetime.datetime.now()
    ents = []
    for i, part in enumerate(s):
        if not part:
            continue
        elif len(part) <= 36:
            part = f"{part} . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . ."
        curr = []
        try:
            # print(yellow(f"[ model_predict ]"), " :: Next input:")
            # for line in textwrap.wrap(part):
            #     print("                ", cyan(line))

            results = allennlp_model.predict(sentence=part)
            # print(yellow(f"[ model_predict ]"), green(" :: OK"))
        except Exception as e:
            print(
                yellow(f"[ model_predict ]"), red(f" :: {e.__class__.__name__}! :: {e}")
            )
            continue
        for word, tag in zip(results["words"], results["tags"]):
            if not re.search(r"(LOC)", tag):
                continue
            elif word.startswith("'") and curr:
                curr[-1] += word
            else:
                curr.append(word)
            if tag[0] in "LU":
                span = " ".join(curr)
                if len(span) >= 3:
                    ents.append(span)
                curr = []
    finish = datetime.datetime.now()
    elapsed = (finish - start).total_seconds()
    mins, secs = elapsed // 60, elapsed % 60
    human_readable = (
        f"{magenta(str(int(mins)).zfill(2))}m {blue(str(int(secs)).zfill(2))}s"
    )
    # print(json.dumps(results))
    print(yellow(f"[ model_predict ]"), f" :: Extraction complete.")
    print(yellow(f"[ model_predict ]"), f" :: Elapsed time : ", human_readable)

    print(
        "======================================================================================================"
    )
    print(green(s))
    print(
        "======================================================================================================"
    )
    cased = defaultdict(list)
    for ent in ents:
        cased[ent.lower()].append(ent)
    for k, v in cased.items():
        cased[k] = list(sorted(v, key=lambda x: v.count(x)))
    freqs = {v[-1]: len(v) for v in sorted(list(cased.values()), key=len, reverse=True)}
    print(blue("Extracted entities:"))
    print(cyan(json.dumps(freqs, indent=4)))
    return freqs


if __name__ == "__main__":
    updates = []
    print(f"Extracting entities..")
    # ner model sometimes fails to recognize newlines as token span boundaries, which they almost always are. to mitigate this,
    # we'll insert a sequence of null tokens between line starts and line ends
    pad = "\n . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . \n"
    queue = list(sorted(list(crawldb.find(prediction="approved", ner=None)), key=lambda x: x['published_at']))
    print(f"Queued {len(queue)} articles")
    for row in queue:
        if row['ner']:
            continue
        print(f"Updating {row}")
        # content = pad.join(
        #     [
        #         re.sub(r"(\s*\n+\s*)", pad, str(attr))
        #         for attr in [
        #             row["title"],
        #             row["loc"],
        #             row["state"],
        #             row["description"],
        #             row["content"],
        #         ]
        #         if attr
        #     ]
        # )
        summary = row['description'] if row['description'] else ''
        title = row['title'] if row['title'] else ''
        content = re.split("\s+", row['content']) if row['content'] else []
        content = ' '.join(content[0:min(len(content),300)])
        counts = extract_entities_with_allennlp(
            title, summary, *content.split("\n")
        )
        updates.append({"url": row['url'], "ner": counts})
        print(
            magenta(
                f"Processing complete. ( {len(updates)} / 50 ) rows queued for insertion."
            )
        )
        if len(updates) > 50:
            print(f"Inserting 50 rows...")
            crawldb.update_many(updates, ["url"])
            updates = []
    crawldb.update_many(updates, ["url"])
