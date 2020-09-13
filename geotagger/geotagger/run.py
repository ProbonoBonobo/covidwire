from textdistance import JaroWinkler
from shapely.geometry import Polygon, MultiPolygon, Point
from munch import Munch
import os
import numpy as np
import random
from gemeinsprache.utils import green, yellow, blue, cyan, red, magenta
import googlemaps
from collections import defaultdict
from math import sqrt, log
from textwrap import wrap
from geopy.distance import geodesic
import pandas as pd
from geopy.distance import geodesic
from urllib.request import urlopen
import json
import plotly.io as pio
from urllib.request import urlopen
import json
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import ktrain
predictor = ktrain.load_predictor('/home/kz/dev/model.bin')
pio.renderers.default = "browser"
import re
import dataset
import psycopg2
import os

GENERATE_PLOTS = False
TABLE = 'articles_v2'
# #
db_config = {
    "user": "kz",
    "password": "admin",
    "host": "127.0.0.1",
    "port": "5432",
    "database": "cvwire",
}
local_db = dataset.connect(
    f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
)

# Production DB
db_config = {
    "user": "admin",
    "password": os.environ.get("MODERATION_PASSWORD", "Feigenbum4"),
    "host": "64.225.121.255",
    "port": "5432",
    "database": "covidwire",
}

db = dataset.connect(
    f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
)
conn = psycopg2.connect(**db_config)

import json
from hashlib import blake2b
import re
from gemeinsprache.utils import blue, red
from gemeinsprache.utils import blue, green, cyan, yellow, magenta

nlp = None
allennlp_model = None
query_rewrite_rules = {
    "U.S.": "United States",
    "U.S" : "United States",
    "US": "United States",
    "Wall Street": "Manhattan, New York",
    "White House": "Washington, D.C.",
    "UC San Diego": "University of California, San Diego",
    "UCSD": "University of California, San Diego"
}

def deg2dec(coord):
    patt = re.compile(
        r"(?P<deg>[\d\.]{0,3})°\s?(?P<m>[\d\.]{0,6})?′?\s?(?P<s>[\d\.]{0,8})?[^\w]?(?P<dir>[NWSE])"
    )
    lat, lon = [re.search(patt, c).groupdict() for c in coord]
    lat["m"] = float(lat["m"]) if lat["m"] else 0
    lon["m"] = float(lon["m"]) if lon["m"] else 0
    lat["s"] = float(lat["s"]) if lat["s"] else 0
    lon["s"] = float(lon["s"]) if lon["s"] else 0

    if lat["s"]:
        lat["m"] = lat["m"] + (lat["s"] / 60)
    if lon["s"]:
        lon["m"] = lon["m"] + (lon["s"] / 60)
    lat_m = lat["m"] / 60
    lon_m = lon["m"] / 60
    lat = (int(lat["deg"]) + lat_m) * [1, -1][int(lat["dir"] in "WS")]
    lon = (int(lon["deg"]) + lon_m) * [1, -1][int(lon["dir"] in "WS")]
    return lon, lat


def blake(thing):
    """Calculate a blake2b hash value for any valid object. If `thing` isn't a string, check to see if it has a __dict__
       representation that could be hashed instead (so different references to the same value will hash to the same
       value); otherwise, use its __repr__() value as a fallback."""
    thingstring = (
        thing
        if isinstance(thing, str)
        else repr(thing.__dict__)
        if hasattr(thing, "__dict__")
        else repr(thing)
    )
    return blake2b(thingstring.encode("utf-8")).hexdigest()


import inspect
from itertools import zip_longest


def serialize_call_args(f):
    argspec = inspect.getfullargspec(f)
    argnames = argspec.args
    n_argnames = len(argnames) if argnames else 0
    n_defaults = len(argspec.defaults) if argspec.defaults else 0
    print(n_argnames, n_defaults)
    default_vals = [None for i in range(n_argnames - n_defaults)]
    if n_defaults:
        default_vals += list(argspec.defaults)

    def wrapped(*args, **kwargs):
        argmap = {}
        for k, arg, default in zip_longest(argnames, args, default_vals):
            argmap[k] = arg if arg else kwargs[k] if k in kwargs else default
        return argmap

    return wrapped


import dill


def cache_queries(func):
    sym = func.__name__
    cachepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache.dill")
    argmapper = serialize_call_args(func)

    def wrapped(*args, **kwargs):
        if 'cache' not in globals():
            global cache
            try:
                with open(".cache.dill", "rb") as f:
                    cache = dill.load(f)
            except:
                cache = {}
                with open(".cache.dill", "wb") as f:
                    dill.dump(cache, f)
        if sym not in cache:
            cache[sym] = {}
        call_args = argmapper(*args, **kwargs)
        call_args["op"] = sym
        hashable = blake(call_args)
        if "cache_override" in kwargs and kwargs["cache_override"]:
            print(f"Overriding cached query: {args}")
        elif hashable in cache[sym]:
            print(f"Reusing cached query: {hashable}")
            return cache[sym][hashable]["__output__"]
        else:
            print(f"Cache miss. Executing API query for: {hashable}")
        out = func(*args, **kwargs)
        call_args["__output__"] = out
        cache[sym][hashable] = call_args
        if not random.randint(0, 150):
            with open(".cache.dill", "wb") as f:
                dill.dump(cache, f)
        return cache[sym][hashable]["__output__"]

    return wrapped


with urlopen(
    "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
) as response:
    plotly_data = json.load(response)
# API_KEY = os.environ['GOOGLE_MAPS_API_KEY']

crawldb = db[TABLE]
shapedb = db["geojson"]
counties = {}
states = {}
county2fips = {}
for row in shapedb:
    coords = eval(row["hull"].replace("{", "[").replace("}", "]"))
    county = row["name"]
    hull = Polygon(coords)
    counties[county] = hull
    county2fips[county] = row["fips"]
for row in db['uscities']:
    county = f"{row['county_name']} County, {row['state_name']}"
    fips = row['county_fips']
    county2fips[county] = fips
# gmaps = googlemaps.Client(key=os.environ["GOOGLE_MAPS_API_KEY"])
rows = []
bias = {}

from pyproj import Geod


def calculate_bounding_box(point, km):
    lon, lat = point
    geod = Geod(ellps="WGS84")
    ne1, ne2, az1 = geod.fwd(lon, lat, az=45, dist=km / 2 * 1000, radians=False)
    sw1, se2, az2 = geod.fwd(lon, lat, az=az1, dist=km / 2 * 1000, radians=False)
    c1 = (ne1, ne2)
    c2 = (sw1, se2)
    return c1, c2




def diag2poly(p1, p2):
    points = Polygon([p1, [p1[0], p2[1]], p2, [p2[0], p1[1]]])
    return points

#
# @cache_queries
# def geocode(*args, **kwargs):
#     globals()["geocode"].__doc__ = gmaps.geocode.__doc__
#     result = gmaps.geocode(sourceloc)
#     return result
#
#
# @cache_queries
# def find_place(*args, **kwargs):
#     globals()["find_place"].__doc__ = gmaps.find_place.__doc__
#     result = gmaps.find_place(*args, **kwargs)
#     return result
#
#
# def get_bounding_box(s, poi=None):
#     match_types = set()
#     """
#     for result in hypdb.find(name=s):
#         match_types.update([m for m in [result['hypernym'],  result['extension']] if m])
#     qualnames = [row['qualname'] for row in hypdb if s in row['qualname']]
#
#     geotypes = ("Settlement", "District", "Community", "Neighborhood", "City", "Capitol", "State", "Country", "Valley", "Mountain")
#     is_geo = len(qualnames) >= 10 or any(t in match_types for t in geotypes)
#     """
#
#     def get_locationbias(sourceloc):
#         response = geocode(sourceloc)
#         lat, lon = list(response[0]["geometry"]["location"].values())
#         querystring = f"point:{lat},{lon}"
#         bias[sourceloc] = querystring
#         return querystring
#
#     location_bias = get_locationbias(poi) if poi else None
#     response = find_place(
#         s,
#         location_bias=location_bias,
#         input_type="textquery",
#         fields=["name", "geometry", "formatted_address"],
#     )
#     if poi and "California" not in poi:
#         response["candidates"] = [
#             candidate
#             for candidate in response["candidates"]
#             if geodesic(my_loc, list(candidate["geometry"]["location"].values())).km
#             > 250
#         ]
#     err = None
#     try:
#         candidate = response["candidates"][0]
#         bb = [
#             list(sorted(list(p.values())))
#             for p in candidate["geometry"]["viewport"].values()
#         ]
#         coords = diag2poly(*bb)
#         poly = Polygon(coords)
#         diameter = geodesic(list(reversed(bb[1])), list(reversed(bb[0]))).kilometers / 2
#
#         centroid = poly.centroid
#         bb = calculate_bounding_box(centroid.coords[0], diameter * 0.38)
#         area = poly.area
#         name = response["candidates"][0]["name"]
#     except Exception as e:
#         print(e.__class__.__name__, e)
#         bb = []
#         coords = []
#         poly = Polygon([[0, 0], [0, 0], [0, 0], [0, 0]])
#         centroid = poly.centroid
#         area = poly.area
#         err = e
#         name = None
#     return {
#         "bias": location_bias,
#         "response": response,
#         "name": name,
#         "bounds": bb,
#         "poly": poly,
#         "centroid": centroid,
#         "area": area,
#         "err": err,
#     }


def visualize(df, title, subtitle, center, article_id, content):
    import plotly.express as px

    mapbox_access_token = "pk.eyJ1IjoibmVvbmNvbnRyYWlscyIsImEiOiJjazhzazZxNmQwaG4xM2xtenB2YmZiaDQ5In0.CJhvMwotvbdJX4FhbyFCxA"
    import plotly.graph_objects as go

    fig = go.Figure(
        go.Choroplethmapbox(
            geojson=plotly_data,
            locations=df.fips,
            z=df.z,
            name="county",
            hoverinfo="all",
            text=[
                f"County: {county} <br>Tokens: {x} <br>Score: {score}"
                for x, score, county in zip(df.tokens, df.z, df.county)
            ],
            marker_line_width=0,
            marker_opacity=[max(z / max(df.z), 0.1) for z in df.z],
            zmin=0,
            zmax=max(6, max(df.z)),
        )
    )
    fig.update_layout(
        mapbox_style="dark",
        mapbox_accesstoken=mapbox_access_token,
        title=title,
        paper_bgcolor="darkgray",
        plot_bgcolor="darkgray",
        mapbox_zoom=8,
        mapbox_center=center,
    )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.show()
    # os.makedirs(f"outputs/{article_id}", exist_ok=True)
    # with open(f"outputs/{article_id}/content.txt", "w") as f:
    #     f.write(content)
    # fig.write_html(f"outputs/{article_id}/fig.html")


#
jaro_winkler = JaroWinkler()


import shapely


def load_geojson(feature):
    t = feature["type"]

    coords = feature["coordinates"]
    print(t, coords)
    constructor = getattr(shapely.geometry, t)
    if t == "MultiPolygon":
        shape = constructor([shapely.geometry.Polygon(x) for x in coords])

    else:
        shape = constructor(coords[0])
    return shape


county_shapes = {}
state_shapes = {}
for i, row in enumerate(db["geojson_v4"]):
    cf = (red, yellow, green, blue, cyan, magenta)[i % 6]
    target = county_shapes if row["feature_type"] == "County" else state_shapes

    try:
        target[row["qualname"]] = load_geojson(row["geometry"])
    except:
        target[row["qualname"]] = load_geojson(row["hull"])

batch = []
queue = list(
    sorted(
        [row for row in db.query("select *, docvec_v2[2] -  docvec_v2[1] as score from articles_v2 where prediction = 'approved' and published_at is not null and scoring_version not like 'v6:%' limit 5000;") if row['published_at'] and row['docvec_v2'] and len(row['docvec_v2']) == 8],
        key=lambda x: x["published_at"],
        reverse=True,
    )
)
print(f"{len(queue)} items in queue")
my_loc = [float(os.getenv("LAT", 32.74)), float(os.getenv("LONG", -117.13))]
labels = predictor.get_classes()
import textwrap
def predict_location(article):
  #lines = [k.title() + ": " + fix_text(re.sub(r"\s+", " ", v.replace("\n", " "), re.MULTILINE)) +  " " for k,v in article.items()]
  lines = article.split("\n")
  stub = article
  probs = predictor.predict(stub, return_proba=True)
  srted = [(k, round(v*100,2)) for k,v in sorted(list(zip(labels,probs.tolist())), key=lambda x: x[1], reverse=True)]
  prediction, confidence = srted[0]

  filtered = {k:v for k,v in filter(lambda x: x[1]>0.5, srted)}
  colors = [red, yellow, green, blue]
  for color, line in zip(colors, lines):
    for wrapped in textwrap.wrap(line):
      print(color(wrapped))
  print("="*20, "RANKED PREDICTION", "="*20, "\n")
  print(json.dumps(filtered, indent=4), end="\n\n" + ("=" * 59) + "\n\n")
  return {"prediction": prediction, "confidence": confidence, "output": filtered}
from shapely.ops import cascaded_union

base_scores = {"indefinite": 2, "international": 3, "national": 5, "regional": 8, "state": 13, "city": 21}
import math
for row in queue:
    # if not row["ner"]:
    #     continue

    #
    # print(row)
    words = re.findall(r"\b(\S+)\b", row['content'], re.MULTILINE)
    rejection_pct,approval_pct = row['docvec_v2'][:2]
    approval_score = (min(row['score'], 5) - abs(4.666666666 - min(row['score'], 5))) * (1/4.666666666 )

    approved = approval_pct > rejection_pct
    preview = ' '.join(words[:min(len(words), 150)])
    entry = f"""Headline: {row["title"]} \nSource: {row['name']} \nDescription: {row["description"]} \nPreview: {preview}"""
    chunked = "\n".join(map(str, entry))
    tagged = predict_location(entry)
    prediction = tagged['prediction']
    confidence = tagged['confidence']
    outputs = tagged['output']
    audience = row['audience']
    high_quality_sources = {"The Atlantic", "Los Angeles Times", "The Washington Post", "The Philadelphia Inquirer", "Reason Magazine", "KPBS, San Diego", "Mother Jones", "The Center for Public Integrity", "The Boston Globe",  "New York Magazine", "New York Times", "Pro Publica", "ProPublica", "Columbia Journalism Review", "The Economist", "Washington Post", "Wired", "The Boston Globe", "Wall Street Journal", "San Francisco Chronicle"}
    is_high_quality_source = row['name']
    if not audience:
        continue

    # if prediction == 'United States':
    #     fips = ''
    # elif prediction.lower() in ('international', 'unspecified'):
    #     fips = prediction
    #     continue
    # else:
    shape=None
    if prediction in county_shapes:
        shape = county_shapes[prediction]
    elif prediction in state_shapes:
        shape = state_shapes[prediction]
    elif prediction.startswith("District of Columbia"):
        states = state_shapes.values()
        shape = cascaded_union(list(states)).convex_hull
        print(f"No shape for prediction: {prediction}")

    # else:
    #     print(f"No shape for prediction: {prediction}")
    #     continue
    print(green(prediction), blue(shape))

    total_points = base_scores[audience]
    base = approval_score
    coeff = min(1.0, confidence / 100 + 0.2)
    adjusted = base * coeff * total_points

    print(f"{cyan(row['title'])}")
    print(f"Predicted locale: {magenta(prediction)}")
    print(f"Predicted audience: {magenta(row['audience'])}")
    print(f"Original score: {magenta(row['score'])}")
    print(f"Possible points: {red(total_points)}")
    print(f"Raw score: {yellow(approval_score)}")
    print(f"Geoconfidence: {green(confidence / 100)}")
    print(f"Perplexity penalty: {blue(coeff)}")
    print(f"Percent of total points awarded: {cyan(base * coeff)}")
    print(f"New score: {magenta(adjusted)}")
    # continue
    acc = {k: 0 for k in county_shapes.keys()}
    for county, hull in county_shapes.items():
        if shape:
            c1 = list(sorted(list(*hull.centroid.coords), reverse=True))
            c2 = list(sorted(list(*shape.centroid.coords), reverse=True))
        #
            distance_from_target = (
                1
                if hull.contains(shape) or shape.contains(hull) or hull.almost_equals(shape)
                else geodesic(c1, c2).km
            )
        else:
            distance_from_target = 9999
        coreference_score = 0
        if audience == 'regional':
            distance_penalty = 1-max(math.sqrt(distance_from_target*0.2)/2,1)
            coreference_score = adjusted if distance_from_target <= 1 else adjusted * distance_penalty
        elif audience == 'national':
            quality = 2 if is_high_quality_source and approval_score > 1 and row['docvec_v2'][6] > 4 else 0.9
            coreference_score = min(8, adjusted * quality)
        elif audience == 'city':
            coreference_score = adjusted if county == prediction else 0
        elif audience == 'state':
            my_state = prediction.split(" County, ")[-1]
            same_state = county.endswith(my_state)
            coreference_score = adjusted if same_state else 0
        elif audience == 'international':
            quality = 1.6 if is_high_quality_source and approval_score > 1 and row['docvec_v2'][3] > 4 else 0.9
            coreference_score = min(5, adjusted * quality)
        elif audience == 'indefinite':
            coreference_score = 0
        else:
            coreference_score = 0

        #print(f"County {county} is {distance_from_target}km from {prediction}")
        # coreference_score = 0 if county not in outputs else adjusted if distance_from_target < 5 else adjusted * (outputs[county ]/ adjusted )
        acc[county] += coreference_score
    acc = {k: v for k, v in acc.items()}
    # print(acc)
    local_arr = np.array(list(acc.values()))

    # print(f"=================== Total scores: ====================")
    # print("                 ", cyan(row["title"]), f" ({row['name']}, {row['loc']})")
    # print("                 ", blue(row["description"]))
    #
    # for line in wrap(row["content"]):
    #     print(yellow(line),)
    # ranked = list(sorted(list(acc.items()), key=lambda x: x[1], reverse=True))
    # for county, score in sorted(list(acc.items()), key=lambda x: x[1], reverse=True):
    #     if score >= 0.2:
    #
    #         print(
    #             f"{green(county):<36} :: {blue(score)}"
    #         )
    db['articles_v2'].update({
            "url": row['url'],
            "predicted_location_name": prediction,
            "predicted_location_confidence": confidence,
            "scoring_version": "v6:ktrain-geoclassifier-smooth-regional-falloff",
            "scored": {county2fips[k]: v for k, v in acc.items() if k in county2fips},
        }, ['url'])





    #
    #             if hull.intersects(poly):
#
#     ents = row["ner"]
#     sourceloc = row["loc"]
#     state = row["state"]
#     resolved_entities = {}
#     resolved_names = defaultdict(list)
#     if not ents:
#         continue
#     acc = {k: 0 for k in county2fips.keys()}
#     for ent, weight in ents.items():
#         if ent in query_rewrite_rules:
#             ent = query_rewrite_rules[ent]
#
#         result = Munch(get_bounding_box(ent, sourceloc))
#         resolved_entities[ent] = result.name
#         location_bias = result.bias
#         print(f"Location bias: {location_bias}")
#         origin_coords = [float(x) for x in location_bias.split(":")[-1].split(",")]
#         origin = Point(origin_coords)
#
#         # print(result)
#         # print(result.response)
#         if result.err:
#             print(result.err)
#             continue
#         poly = result.poly if ent not in state_shapes else state_shapes[ent]
#         difference_penalty = pow(
#             jaro_winkler.normalized_similarity(result.name, ent), 3
#         )
#         for county, hull in counties.items():
#
#             if hull.intersects(poly):
#
#                 c1 = list(sorted(list(*hull.centroid.coords), reverse=True))
#                 c2 = list(sorted(list(*result.centroid.coords), reverse=True))
#
#                 distance_from_target = (
#                     1
#                     if hull.contains(poly) or poly.contains(hull) or hull.almost_equals(poly)
#                     else 1 / sqrt((1 + geodesic(c1, c2).km) / 4)
#                 )
#                 # distance_from_source = origin.distance(hull)
#                 distance_penalty = distance_from_target
#
#                 size_penalty = pow(1 + result.area, 2) / 6
#                 unweighted_score = log(1 + pow(1 + weight, 2), 2)
#                 relevance_score = (
#                     sqrt(unweighted_score / size_penalty)
#                     * distance_penalty
#                     * difference_penalty
#                 )
#                 resolved_names[county].append(f"{ent} ( => {result.name} )")
#
#                 #
#                 # print(f"County {green(county)} intersects {blue(result.name)} with score: {magenta(relevance_score)}")
#                 # print(f"    relevance_score = sqrt(unweighted_score / size_penalty) * distance_penalty * difference_penalty")
#                 # print(red(f"    = sqrt({round(unweighted_score)} / {round(size_penalty,8)}) * {distance_penalty} * {difference_penalty}") )
#                 # print(yellow(f"    = sqrt({(unweighted_score/size_penalty) * distance_penalty } * {difference_penalty}") )
#                 # print(green(f"    = {relevance_score}"))
#                 acc[county] += relevance_score
#     acc = {k: v for k, v in acc.items()}
#     local_arr = np.array(list(acc.values()))
#     if local_arr.max() >= 20:
#         scalar_coeff = 20 / local_arr.max()
#         local_arr = local_arr * scalar_coeff
#         acc = dict(zip(acc.keys(), local_arr))
#     # if local_sos > global_sos:
#     #     scalar_ratio = local_sos / global_sos
#     #     print(cyan(f"Rescaling by constant factor of {scalar_ratio} (global mean: {global_sos}; local mean: {local_sos})"))
#     #     acc = dict(zip(acc.keys(), local_arr * scalar_ratio))
#     # else:
#     #     scalar_ratio = 1
#     # cumsums.append(local_sos)
#     # global_sos = np.array(cumsums).mean()

#
#     # print({k:v for k,v in row['scored'].items() if v > 0})
#     batch.append(row)
#     # for k,v in list(globals().items()):
#     #     print(blue(k), " :: ", magenta(v))
#     # pause = input("Press any key to continue, or :q to quit")
#     # if pause == ':q':
#     #     breakpoint()
#
#     crawldb.update(row, ['url'])
#     if GENERATE_PLOTS:
#         lng, lat = list(counties[ranked[0][0]].centroid.coords)[0]
#         center = {"lat": lat, "lon": lng}
#         struct = pd.DataFrame(
#             [
#                 {
#                     "county": k,
#                     "fips": county2fips[k],
#                     "z": v,
#                     "tokens": ", ".join([f"{x}<br>" for x in resolved_names[k]]),
#                 }
#                 for k, v in acc.items()
#             ]
#         )
#         if max(struct.z) >= 0.2:
#             title = f"[#{row['url']}] {row['title']} ({row['name']}, {row['loc']})"
#             subtitle = {row["description"]}
#             visualize(struct, title, subtitle, center, row["url"], chunked)
# print(len(queue))
