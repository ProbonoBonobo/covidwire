import flask
from flask import request
from api.creds import db_config
from flask_cors import CORS
from scipy.spatial.distance import euclidean, cosine
import datetime
import dataset
import json
import time
import html
from urllib.parse import unquote_plus as urldecode
cache = {}
TABLE = "articles_v2"

db = dataset.connect(
        f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    )
articles = db[TABLE]

app = flask.Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def home():
    defaults = {"_limit": 50}
    kwargs = request.args or {}
    kwargs = dict(**defaults, **request.args)
    formatted_args = '</p><p>'.join([f"{k} = {v}\n" for k,v in request.args.items()])
    results = articles.find(**kwargs)
    formatted_results = '</div><div>'.join([json.dumps(result, indent=4, default=lambda x: x if isinstance(x, (str, int, float, list)) else str(x)).replace(" ", "&nbsp;").replace("\n", "<br>") for result in results])
    return f"""<h1>Query Parameters:</h1><p>{formatted_args}</p>
               <h2>Results:</h2><div>{formatted_results}</div>"""
import random
@app.route('/scored', methods=['GET'])
def get_scored_results():
    print(f"Cache keys: {list(cache.keys())}")
    kwargs = {"_limit": 50, "p": 0, "fips": "06097", "audience": ""}
    _kwargs = {k: urldecode(v) for k,v in request.args.items()} or {}
    kwargs.update(_kwargs)

    fips = kwargs['fips']
    print(f"fips is : {fips}")
    start = int(kwargs['_limit']) * int(kwargs['p'])
    stop = start + int(kwargs['_limit'])
    print(f"Start is: {start}, Stop is: {stop}")
    results = None
    if fips in cache and cache[fips] and cache[fips][0] > time.time():
        results = cache[fips][1]
    else:
        query = f"""SELECT distinct on (title)
        published_at, name, title, description, url, docvec_v2 as docvec, ner, image_url, audience, mod_status, prediction, city, state, loc, cast(scored ->> '{fips}' as float)  / (extract(days from (now() - published_at)) + 1) as score
    FROM
        {TABLE}
    WHERE prediction = 'approved' and scored ->> '{fips}' != '0';"""
        results = []
        for r in sorted(list(db.query(query)), key=lambda x: x['score'], reverse=True):
            # try:
            #     r['docvec'] = list(map(float,r['docvec'][1:-1].split(",")))
            #
            #
            # except:
            #     r['docvec'] = [x*0.00000001 for x in random.sample(range(-70000000,70000000), 8)]
            results.append(r)

        cache[fips] = [time.time() + 3200, results]
    #print(results[start:stop])
    #ormatted_args = '</p><p>'.join([f"{k} = {v}\n" for k, v in request.args.items()])
    #formatted_results = '</div><div>'.join([json.dumps(result, indent=4, default=str).replace(" ", "&nbsp;").replace("\n", "<br>").replace("\\n", "<br>") for result in results[start:stop]])
    if kwargs['audience']:
        audience = kwargs['audience']
        results = [result for result in results if result['audience'] == audience]
    results = results[start:min(stop, len(results))]
    response = app.response_class(
        response = json.dumps(results, indent=4, default=lambda x: x if not isinstance(x, datetime.datetime) else x.isoformat()),
        status=200,
        mimetype = 'application/json')
    return response
    # return f"""<h1>Query Parameters:</h1><p>{formatted_args}</p>
    #                <h2>Results:</h2><div>{formatted_results}</div>"""

@app.route('/classified', methods=['GET'])
def get_classifier_predictions():
    actual_labels = ("approved", "rejected", "international", "city", "regional", "national", "indefinite", "state")
    classifier_labels = ("approved", "rejected", "international", "local", "regional", "national", "unbound", "state")
    transtable = dict(zip(actual_labels, classifier_labels))

    kwargs = {"audience": "local,regional,state,national,international,indefinite",
              "sortOrder": "ambiguity",
              "_limit": 50,
              "p": 0}
    def ambiguousness(indices):
        def inner(row):
            vec = row['docvec_v2']
            ordered = list(sorted([(v, index) for index, v in enumerate(vec)], reverse=True))
            _max, _maxindex = ordered[0]
            if _maxindex in indices:
                _nextmax, _nextmaxindex = ordered[1]
                return abs(_max - _nextmax)
            else:
                return -1
        return inner
    def gradient(indices):
        def inner(row):
            vec = [x for i,x in enumerate(row['docvec_v2']) if i in indices]
            _max = max(vec)
            print(_max)
            return _max
        return inner


    _kwargs = {k: urldecode(v) for k, v in request.args.items()} or {}
    kwargs.update(_kwargs)
    print(kwargs)
    keys = {"ambiguity": ambiguousness, "gradient descent": gradient}
    serialized_kwargs = str({k:v for k,v in kwargs.items() if k not in ('p',)})
    sort_function = keys[kwargs['sortOrder']]
    if serialized_kwargs in cache and cache[serialized_kwargs][0] > time.time():
        results = cache[serialized_kwargs][1]
    else:
        selected_labels = set(kwargs['audience'].split(","))
        if not all(label in classifier_labels for label in selected_labels):
            response = app.response_class( response = "invalid audience: {selected_labels} valid audiences: {classifier_labels}", status=200)
            return response
        docvec_indices = set([classifier_labels.index(label) for label in selected_labels])
        # print(f"Selected labels: {selected_labels} Indices: {docvec_indices}")
        filtered = []
        for article in db.query("select distinct on (title) name, loc, title, description, url, published_at, image_url, content, docvec_v2, audience, prediction from articles_v2 where docvec_v2 is not null;"):
            if article['audience'] in transtable and transtable[article['audience']] in selected_labels:
                article['docvec_v2'] = dict(zip(classifier_labels, article['docvec_v2']))
                filtered.append(article)

        results = list(sorted(filtered, key=sort_function(docvec_indices), reverse=True))[0:max(len(filtered), kwargs['_limit'])]
        cache[serialized_kwargs] = (time.time() + 3600, results)
        # print(f"Ordered: {ordered}")
    results = results[int(kwargs['p'])]
    response = app.response_class(
        response = json.dumps({"results": results}, indent=4, default=lambda x: x if not isinstance(x, datetime.datetime) else x.isoformat()),
        status = 200,
        mimetype='application/json')
    print(response)
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888)

