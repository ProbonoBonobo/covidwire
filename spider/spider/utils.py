import json
from gemeinsprache.utils import yellow, green
from flatdict import FlatterDict
from ftfy import fix_text_segment
import re
from transformers import RobertaForSequenceClassification, RobertaTokenizer, pipelines
from gemeinsprache.utils import magenta, cyan, green, yellow, blue, red


class DotDict(dict):
    """
    a dictionary that supports dot notation
    as well as dictionary access notation
    usage: d = DotDict() or d = DotDict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        for key, value in dct.items():
            if hasattr(value, "keys"):
                value = DotDict(value)
            self[key] = value
        super().__init__()


def load_metadata(soup):
    if not soup:
        return {}
    metas = soup.select("script[type='application/ld+json']")
    o = {"errors": []}
    for meta in metas:
        try:
            _o = json.loads(meta.string)

        except Exception as e:
            o["errors"].append([f"{e.__class__.__name__} :: {e}", meta.string])
            continue

        if isinstance(_o, list):
            for __o in _o:
                o[__o["@type"]] = __o
        elif "@type" in _o and isinstance(_o["@type"], list):
            for t in _o["@type"]:
                for __o in _o:
                    o[t] = __o
        elif "@type" in _o and _o["@type"] in o:
            o[_o["@type"]] = _o
        else:
            continue

    flat = FlatterDict(o)
    o = {}
    for k, v in flat.items():

        if (
            isinstance(v, str)
            and not re.search(r"(image|url)", k)
            and not re.match(r"\s*http", v)
        ):
            txt = fix_text_segment(v)
            if txt and "<" in txt:
                txt = re.sub(r"<[^>]*>", "", txt)
            o[k] = txt
        else:
            o[k] = v
    flat = json.loads(
        json.dumps(
            {k: v for k, v in o.items()},
            indent=4,
            default=lambda x: dict(**x) if isinstance(x, FlatterDict) else x,
        )
    )
    print(flat)
    return flat


class Haystack(dict):
    def __init__(self, soup):
        super().__init__(load_metadata(soup))

    def search(self, attrs: list, atoms_only=True):
        default = None
        for k in attrs:
            if k in self:
                try:
                    v = self.__getitem__(k)
                except Exception as e:
                    print(e.__class__.__name__, e, k)
                    continue
                if not v:
                    continue
                elif atoms_only and isinstance(v, (str, int, float)):
                    return v
                elif not atoms_only:
                    return v
        return None

    def re_search(self, regex_attrs, atoms_only=True):
        for k1 in regex_attrs:
            filtered = {k2: v2 for k2, v2 in self.items() if re.search(k1, k2)}
            if filtered:
                return list(filtered.values())
        return []


import numpy as np


class Model:
    def __init__(self, model_dir):
        self.model = RobertaForSequenceClassification(
            model_dir, output_attentions=True, output_hidden_states=True
        )
        self.tokenizer = RobertaTokenizer(
            model_dir,
            add_special_tokens=True,
            merges_file=os.path.join(model_dir, "merges.txt"),
        )

    def encode(self, sent):
        tokenized = self.tokenizer.encode(sent, return_tensors="pt")
        classifier, attentions, hidden_states = self.model(tokenized)
        return {repr(k): k for k in (classifier, attentions, hidden_states)}

    def sent2vec(self, sent):
        encoded = self.encode(sent)
        tensors = encoded["hidden_states"]
        mean_vec = np.array(
            [np.array(t[0].median(0).values.detach().numpy()) for t in tensors]
        ).mean(0)


def get_selector(child):
    path = list(reversed([node.name for node in child.parentGenerator()][:-1]))
    if hasattr(child, "name") and child.name:
        path.append(child.name)
    return " ".join(path)


def elect_best_selector(soup):
    url = soup.url
    candidates = []
    votes = []
    maybe_roots = (
        ".article-content",
        ".articleBody",
        ".contentAccess",
        ".article-body",
        "#story",
        "#articleBody",
        "#article-body",
        "#article-content",
        "#article",
        "article",
    )
    root = soup.select("body")
    for slx in maybe_roots:
        if soup.select(slx):
            root = soup.select(slx)
            # print(cyan(f"using selector {slx} for url {url}"))
            votes = [slx]
    for node in root:
        for child in node.recursiveChildGenerator():
            try:
                if child.name in ("p"):
                    slx = get_selector(child)
                    classes = set()
                    for parent in child.parentGenerator():
                        if "class" in parent.attrs and parent.attrs["class"]:
                            classes.update(parent.attrs["class"])
                    if "class" in child.attrs and child.attrs["class"]:
                        classes.update(child.attrs["class"])
                    bad_classes = [
                        c
                        for c in classes
                        if re.search(
                            r"(comment|social|alert|promo|advertis)", c, re.IGNORECASE
                        )
                    ]
                    if not bad_classes:

                        candidates.append(
                            (re.sub(r"\s+", " ", child.text.strip()), slx)
                        )
                    else:
                        print(
                            magenta(
                                f"Skipping {child.name} node with classes {bad_classes}"
                            )
                        )
            except:
                continue
    candidates = list(sorted(candidates, key=lambda x: len(x[0])))
    # print(yellow(url))
    for i, node in enumerate(candidates[-5:]):
        u, slx = node
        votes.append(slx)
        # print(blue(i), green(u), len(u))
    votes = list(sorted(votes, key=lambda x: votes.count(x), reverse=True))
    if not votes:
        votes = ['p']
    winner = votes[0]
    content = []
    selected = soup.select(votes[0])
    for node in selected:
        for child in node.recursiveChildGenerator():
            try:
                if child and child.string and len(child.string.strip()) > 36:
                    content.append(child.string.strip())
            except Exception as e:
                continue
    content = "\n".join(content)
    # print("=" * 80)
    # print(red(winner))
    # print(green(content))
    # print("=" * 80)
    return winner, content

def extract_content(soup, slx):
     smell = re.compile(r"(error|email|Facebook|Twitter|signup|your request|delivered every|let us know|plus:|" +
                          r"by visiting|mailing list|follow him|follow her|columnist|your inbox|" +
                          r"this story|contributed to|this site|to view|download|sorry,|this article" +
                          r"|subscri|©|copyright|@|sign up|your opinion|share this|share on|read more" +
                          r"|follow us|click here|this report|editor|staff writer|is a reporter|contact us|" +
                          r"related:|log in|more:)", re.IGNORECASE)

     selected = soup.select(slx)
     content = []
     for node in selected:
         inherited_classes = set()
         txt = node.text.strip()
         if len(txt) >= 30 and re.search(r"[.\"?)”]$", txt):
             for parent in node.parentGenerator():
                 try:
                     inherited_classes.update(parent.attrs['class'])
                 except:
                     continue
             txt = re.sub(r"\s+", " ", txt)
             if re.search(smell, txt):
                 matches = re.findall(smell, txt)
                 print(f"Ignoring {red(txt)}")
                 print(f"    Matches: ", end="")
                 for m in matches:
                     print(f"            {yellow(m)}")
                 continue
             elif txt not in content:
                 bad_classes = [c for c in inherited_classes if re.search(r"(comment|promo|advertis|social)", c)]
                 if not bad_classes:

                     content.append(txt)
                 else:
                     print(f"Skipping due to bad classes: {red(bad_classes)}")
                     print(f"    skipped text: {yellow(txt)}")
     content = '\n'.join(content)
     return content

