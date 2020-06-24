#!/bin/bash
if test -f "~/covidwire/spider/spider.lock"; then
	exit 0
else
	touch ~/covidwire/spider/spider.lock
	exec /root/.pyenv/shims/poetry run python crawl_sitemaps.py > ~/covidwire/spider/crawl_sitemaps.log 
	exec /root/.pyenv/shims/poetry run python parse_responses.py > ~/covidwire/spider/parse_responses.log
	exec /root/.pyenv/shims/poetry run python process_queue.py > ~/covidwire/spider/process_queue.log
	rm ~/covidwire/spider/spider.lock
fi
