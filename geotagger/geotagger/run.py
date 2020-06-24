from rich import print
from textdistance import JaroWinkler
from shapely.geometry import Polygon, MultiPolygon, Point
from munch import Munch
import os
import numpy as np
from gemeinsprache.utils import green, yellow, blue, cyan, red, magenta
import googlemaps
from collections import defaultdict
from math import sqrt, log
from textwrap import wrap
from geopy.distance import geodesic
import pandas as pd
from urllib.request import urlopen
import json
import plotly.io as pio
pio.renderers.default = 'browser'
from urllib.request import urlopen
import json
import plotly.io as pio
pio.renderers.default = 'browser'
import re
import dataset
import psycopg2
import os

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


#Production DB
db_config = {"user": "postgres",
             "password": os.environ.get('MODERATION_PASSWORD', "Feigenbum4"),
             "host": "35.188.134.37",
             "port": "5432",
             "database": "postgres"}

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

    argmapper = serialize_call_args(func)

    def wrapped(*args, **kwargs):
        cache = {}
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
        call_args['op'] = sym
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
        with open(".cache.dill", "wb") as f:
            dill.dump(cache, f)
        return cache[sym][hashable]["__output__"]

    return wrapped



with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    plotly_data = json.load(response)
# API_KEY = os.environ['GOOGLE_MAPS_API_KEY']

crawldb = db['articles']
shapedb = db['geojson']
counties = {}
county2fips = {}
for row in shapedb:
    coords = eval(row['hull'].replace("{","[").replace("}", "]"))
    county = row['name']
    hull = Polygon(coords)
    counties[county] = hull
    county2fips[county] = row['fips']

gmaps = googlemaps.Client(key='AIzaSyBgKU4uNBEw1pmNZ5Fv-Y06tpBqP5TpXpM')
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

blacklisted_ents = ("Wall St.", "Congress", 'The City')


def diag2poly(p1, p2):
    points = Polygon([p1, [p1[0], p2[1]], p2, [p2[0], p1[1]]])
    return points
@cache_queries
def get_bounding_box(s, poi=None):
    match_types = set()
    """
    for result in hypdb.find(name=s):
        match_types.update([m for m in [result['hypernym'],  result['extension']] if m])
    qualnames = [row['qualname'] for row in hypdb if s in row['qualname']]

    geotypes = ("Settlement", "District", "Community", "Neighborhood", "City", "Capitol", "State", "Country", "Valley", "Mountain")
    is_geo = len(qualnames) >= 10 or any(t in match_types for t in geotypes)
    """

    @cache_queries
    def get_locationbias(sourceloc):
        if sourceloc in bias and bias[sourceloc]:
            return bias[sourceloc]
        else:
            response = gmaps.geocode(sourceloc)
            lat, lon = list(response[0]['geometry']['location'].values())
        querystring = f"circle:5000@{lat},{lon}"
        bias[sourceloc] = querystring
        return querystring
    location_bias = get_locationbias(poi) if poi else None
    response = gmaps.find_place(s, location_bias=location_bias, input_type='textquery',
                                fields=['name', 'geometry', 'formatted_address'])
    err = None
    try:
        candidate = response['candidates'][0]
        bb = [list(sorted(list(p.values()))) for p in candidate['geometry']['viewport'].values()]
        coords = diag2poly(*bb)
        poly = Polygon(coords)
        diameter = geodesic(list(reversed(bb[1])), list(reversed(bb[0]))).kilometers / 2

        centroid = poly.centroid
        bb = calculate_bounding_box(centroid.coords[0], diameter * 0.5)
        area = poly.area
        name = response['candidates'][0]['name']
    except Exception as e:
        print(e.__class__.__name__, e)
        bb = []
        coords = []
        poly = Polygon([[0, 0], [0, 0], [0, 0], [0, 0]])
        centroid = poly.centroid
        area = poly.area
        err = e
        name = None
    return {"bias": location_bias, "response": response, "name": name,  "bounds": bb, "poly": poly, "centroid": centroid, "area": area, "err": err}

def visualize(df, title, subtitle, center, article_id, content):
    import plotly.express as px
    mapbox_access_token = "pk.eyJ1IjoibmVvbmNvbnRyYWlscyIsImEiOiJjazhzazZxNmQwaG4xM2xtenB2YmZiaDQ5In0.CJhvMwotvbdJX4FhbyFCxA"
    import plotly.graph_objects as go

    fig = go.Figure(go.Choroplethmapbox(geojson=plotly_data,  locations=df.fips, z=df.z, name='county',

                                         hoverinfo='all', text=[f"County: {county} <br>Tokens: {x} <br>Score: {score}" for x, score, county in zip(df.tokens, df.z, df.county)],
                                        marker_line_width=0,   marker_opacity=[max(z/max(df.z),0.1) for z in df.z],
                                        zmin=0, zmax=max(6, max(df.z)),




                               ))
    fig.update_layout(mapbox_style="dark", mapbox_accesstoken=mapbox_access_token, title=title,
                      paper_bgcolor='darkgray', plot_bgcolor='darkgray',
                      mapbox_zoom=8, mapbox_center=center)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.show()
    # os.makedirs(f"outputs/{article_id}", exist_ok=True)
    # with open(f"outputs/{article_id}/content.txt", "w") as f:
    #     f.write(content)
    # fig.write_html(f"outputs/{article_id}/fig.html")
#
jaro_winkler = JaroWinkler()
cumsums = []
with conn.cursor() as curr:
    curr.execute("select scored from articles where scored is not null;")
    for row in curr.fetchall():
        if row[0]:
            arr = np.array(list(row[0].values()))
            sos = (arr ** 2).sum()
            cumsums.append(sos)

global_sos = np.mean(np.array(cumsums))

batch = []
queue = list(sorted([row for row in crawldb.find(scored=None, mod_status='pending', _limit=10)], key=lambda x: x['published_at'], reverse=True))
for row in queue:
    if not row['ner']:
        continue
    print(row)

    entry = [row['title'], row['description'], row['content']]
    chunked = '\n'.join(map(str, entry))
    ents = row['ner']
    sourceloc = row['loc']
    state = row['state']
    resolved_names = defaultdict(list)
    if not ents:
        continue
    acc = {k: 0 for k in county2fips.keys()}
    for ent, weight in ents.items():

        result = Munch(get_bounding_box(ent, sourceloc))
        location_bias = result.bias
        print(f"Location bias: {location_bias}")
        origin_coords = [float(x) for x in location_bias.split("@")[-1].split(",")]
        origin = Point(origin_coords)


        print(result)
        print(result.response)
        if result.err:
            print(result.err)
            continue
        poly = result.poly
        difference_penalty = pow(jaro_winkler.normalized_similarity(result.name, ent), 2)
        for county, hull in counties.items():
            if hull.intersects(poly):


                distance_from_target = 1 if hull.contains(poly) else pow(10, poly.distance(hull.centroid))
                distance_from_source = origin.distance(hull)
                distance_penalty = distance_from_target + distance_from_source

                size_penalty = pow(1+result.area,2)/6
                unweighted_score = log(1+pow(1+weight, 2), 2)
                relevance_score = sqrt(unweighted_score / distance_from_target / size_penalty) * difference_penalty
                resolved_names[county].append(f"{ent} ( => {result.name} )")


                print(f"County {green(county)} intersects {blue(result.name)} with score: {magenta(relevance_score)}")
                print(f"    relevance_score = sqrt(unweighted_score / distance_from_target / size_penalty) * distance_penalty")
                print(red(f"    = sqrt({round(unweighted_score)} / {round(distance_penalty,8)} / {round(size_penalty,8)}) * {distance_penalty}") )
                print(yellow(f"    = sqrt({round(unweighted_score/distance_penalty/size_penalty, 8)}) * {distance_penalty}") )
                print(green(f"    = {relevance_score}"))
                acc[county] += relevance_score
    acc = {k:v for k,v in acc.items()}
    local_arr = np.array(list(acc.values()))
    if local_arr.max() >= 36:
        scalar_coeff = 36/local_arr.max()
        local_arr = local_arr * scalar_coeff
        acc = dict(zip(acc.keys(), local_arr))
    local_sos = (local_arr ** 2).sum()
    # if local_sos > global_sos:
    #     scalar_ratio = local_sos / global_sos
    #     print(cyan(f"Rescaling by constant factor of {scalar_ratio} (global mean: {global_sos}; local mean: {local_sos})"))
    #     acc = dict(zip(acc.keys(), local_arr * scalar_ratio))
    # else:
    #     scalar_ratio = 1
    # cumsums.append(local_sos)
    # global_sos = np.array(cumsums).mean()


    print(f"=================== Total scores: ====================")
    print("                 ", cyan(row['title']), f" ({row['name']}, {row['loc']})")
    print("                 ", blue(row['description']))

    for line in wrap(row['content']):
        print(yellow(line), )
    ranked = list(sorted(list(acc.items()), key=lambda x: x[1], reverse=True))
    for county, score in sorted(list(acc.items()), key=lambda x: x[1], reverse=True):
        if score >= 0.2:

            print(f"{green(county):<36} :: {blue(score)} {magenta(resolved_names[county])}")
    _row = dict({"ner": ents, "scored": {county2fips[k]: v for k,v in acc.items()}, 'url': row['url']}, **row)
    from dataset.types import JSON
    batch.append(_row)
    # for k,v in list(globals().items()):
    #     print(blue(k), " :: ", magenta(v))
    # pause = input("Press any key to continue, or :q to quit")
    # if pause == ':q':
    #     breakpoint()

    if len(batch) > 5:
        crawldb.update_many(batch, ['url'])
        batch = []
    lng, lat = list(counties[ranked[0][0]].centroid.coords)[0]
    center = {"lat": lat, "lon": lng}
    struct = pd.DataFrame([{"county": k, "fips": county2fips[k], "z": v, "tokens": ', '.join([f"{x}<br>" for x in resolved_names[k]])} for k,v in acc.items()])
    if max(struct.z) >= 0.2:
        title = f"[#{row['id']}] {row['title']} ({row['name']}, {row['loc']})"
        subtitle = {row['description']}
        visualize(struct, title, subtitle, center, row['id'], chunked)






