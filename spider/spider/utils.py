import json
from gemeinsprache.utils import yellow, green
from flatdict import FlatterDict
from ftfy import fix_text_segment
import re


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
        elif isinstance(_o["@type"], list):
            for t in _o["@type"]:
                for __o in _o:
                    o[t] = __o
        else:
            o[_o["@type"]] = _o

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
