from ner.common import init_db
import spacy
from allennlp.predictors import Predictor
import allennlp
import allennlp_models
from collections import defaultdict
import re
import json
from gemeinsprache.utils import red, green, cyan, magenta, blue, yellow

LIMIT_ARTICLES = 100

db = init_db()
crawldb = db["articles"]
allennlp_model = None
import allennlp_models.tagging

predictor = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/ner-model-2020.02.10.tar.gz"
)


def extract_entities_with_allennlp(s):
    model_url = "https://storage.googleapis.com/allennlp-public-models/ner-model-2020.02.10.tar.gz"
    global allennlp_model
    if not "allennlp_model" in globals() or not globals()["allennlp_model"]:
        print(f"Loading AllenNLP NER model...")
        allennlp_model = Predictor.from_path(model_url)
        print(f"Load complete.")
    results = allennlp_model.predict(sentence=s)
    # print(json.dumps(results))
    ents = []
    curr = []
    for word, tag in zip(results["words"], results["tags"]):
        if not re.search(r"(LOC)", tag):
            continue
        elif word.startswith("'"):
            curr[-1] += word
        else:
            curr.append(word)
        if tag[0] in "LU":
            span = " ".join(curr)
            if len(span) >= 3:
                ents.append(span)
            curr = []
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
    freqs = {v[-1]: len(v) for v in cased.values()}
    print(blue("Extracted entities:"))
    print(cyan(json.dumps(freqs, indent=4)))
    return freqs


if __name__ == "__main__":
    updates = []
    print(f"Extracting entities..")
    # ner model sometimes fails to recognize newlines as token span boundaries, which they almost always are. to mitigate this,
    # we'll insert a sequence of null tokens between line starts and line ends
    pad = "\n . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . \n"
    for row in [
        row
        for row in crawldb.find(ner=None, prediction="approved", _limit=LIMIT_ARTICLES)
    ]:
        print(f"Updating {row}")
        content = pad.join(
            [
                re.sub(r"(\s*\n+\s*)", pad, str(attr))
                for attr in [
                    row["title"],
                    row["loc"],
                    row["state"],
                    row["description"],
                    row["content"],
                ]
                if attr
            ]
        )
        counts = extract_entities_with_allennlp(content)
        updates.append({"url": row["url"], "ner": counts})

    crawldb.update_many(updates, ["url"])
