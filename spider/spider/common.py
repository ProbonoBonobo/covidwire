import re
import dataset
import psycopg2
import os
from collections import OrderedDict

target = "prod"
config = {
    "local": {
        "user": "kz",
        "password": os.environ["LOCAL_DB_PASSWORD"],
        "host": "127.0.0.1",
        "port": "5432",
        "database": "cvwire",
    },
    "staging": {
        "user": "postgres",
        "password": os.environ["STAGING_DB_PASSWORD"],
        "host": "34.83.188.109",
        "port": "5432",
        "database": "postgres",
    },
    "prod": {
        "user": "postgres",
        "password": os.environ["PROD_DB_PASSWORD"],
        "host": "35.188.134.37",
        "port": "5432",
        "database": "postgres",
    },
}


def init_db(target=target):
    db_config = config[target]
    return dataset.connect(
        f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    )


def init_conn(target=target):
    db_config = config[target]
    conn = psycopg2.connect(**db_config)
    return conn


default_headers = OrderedDict({
    k.title(): v
    for k, v in {
        "USER-AGENT":
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.113 Safari/537.36",
        "ACCEPT-ENCODING": "gzip, deflate, br",
        "ACCEPT":
        "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        "Connection": "keep-alive",
        "ACCEPT-LANGUAGE": "en-US,en;q=0.9",
        "CACHE-CONTROL": "max-age=0",
        "DEVICE-MEMORY": "8",
        "DNT": "1",
        "DOWNLINK": "10",
        "DPR": "2",
        "ECT": "4g",
        "REFERER": "https://www.google.com/",
        "RTT": "50",
        "SEC-CH-UA": "Google Chrome 81.0.4044.113",
        "SEC-CH-UA-ARCH": "x86_64",
        "SEC-CH-UA-MOBILE": "?0",
        "SEC-CH-UA-MODEL": "",
        "SEC-FETCH-DEST": "document",
        "SEC-FETCH-MODE": "navigate",
        "SEC-FETCH-SITE": "cross-site",
        "SEC-FETCH-USER": "?1",
        "SEC-ORIGIN-POLICY": "0",
        "UPGRADE-INSECURE-REQUESTS": "1",
        "VIEWPORT-WIDTH": "1920",
    }.items()
})


def create_sitemaps_table():
    db = init_db(target)
    if "sitemaps" in db:
        return
    else:
        conn = init_conn(target)
        sql = """
    create table if not exists sitemaps
    (
        url text not null,
        resolved_url text not null,
        site text,
        name text,
        city text,
        state text,
        created_at timestamp default now() not null,
        status_code integer not null,
        status_msg text,
        ok boolean not null,
        encoding text not null,
        bytes integer not null,
        lastmod timestamp,
        response_headers json,
        xmlmeta text,
        content bytea,
        raw bytea,
        error text,
        id bigint,
        loc text
);

"""
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()


def create_articles_table():
    db = init_db(target)
    if "articles" in db:
        return
    conn = init_conn(target)
    sql = """create table articles
(
	id serial not null primary key,
	url text,
	published_at timestamp,
	name text,
	loc text,
	title text,
	description text,
	site text,
	city text,
	state text,
	author text,
	publisher text,
	image_url text,
	content text,
	prediction text,
	mod_status text,
	ner jsonb,
	ner_version text default 'allennlp'::text,
	scored jsonb,
	scoring_version text default 'v1'::text,
	created_at timestamp default now(),
	modified_at timestamp default now(),
	ok boolean
);
"""
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def create_spiderqueue_table():
    db = init_db(target)
    if "articles" in db:
        return
    conn = init_conn(target)
    sql = """create table spiderqueue
(
	url text not null
		constraint spiderqueue_pkey
			primary key,
	site text,
	name text,
	city text not null,
	state text not null,
	loc text not null,
	lastmod timestamp,
	xmlmeta bytea
);
"""
