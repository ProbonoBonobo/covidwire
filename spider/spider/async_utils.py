from typing import List, Iterator
import trio
import httpx
from spider.common import default_headers
import datetime
from urllib.parse import quote_plus
from collections import OrderedDict
import random
import brotli
from ftfy import fix_text_segment


async def fetch_responses(urls: List[str], number_workers: int):
    responses = {}

    async def token_issuer(token_sender: trio.abc.SendChannel, number_tokens: int):
        async with token_sender:
            for _ in range(number_tokens):
                await token_sender.send(None)

    async def worker(url_iterator: Iterator, token_receiver: trio.abc.ReceiveChannel):
        async with token_receiver:
            for url in url_iterator:
                await token_receiver.receive()

                print(f"[{round(trio.current_time(), 2)}] Start loading link: {url}")
                try:
                    async with httpx.AsyncClient() as client:
                        optanon_timestamp = quote_plus(
                            datetime.datetime.now().strftime("%a+%b+%d+%Y+%H:%M:%S")
                        )
                        past = (
                            datetime.datetime.now()
                            + datetime.timedelta(hours=7)
                            - datetime.timedelta(
                                minutes=random.randrange(0, 8),
                                seconds=random.randrange(0, 60),
                            )
                        )
                        optanon_past_timestamp = past.strftime(
                            f"%Y-%m-%dT%H:%M:%S.{random.randrange(0, 999)}Z"
                        )

                        default_cookies = OrderedDict(
                            {
                                "ccpa-state": "No",
                                "OptanonConsent": f"isIABGlobal=true&datestamp={optanon_timestamp}+GMT-0700+(Pacific+Daylight+Time)&version=5.9.0&landingPath=NotLandingPage&groups=C0001%3A1%2CC0002%3A1%2CC0004%3A1&hosts=&geolocation=US%3BCA&AwaitingReconsent=false",
                                "OptanonGlobal": f"isIABGlobal=false&datestamp={optanon_timestamp}+GMT-0700+(Pacific+Daylight+Time)&version=5.15.0&landingPath=NotLandingPage&groups=C0003%3A1%2CC0004%3A1%2CC0005%3A1%2CBG50%3A1%2CC0002%3A1%2CC0001%3A1&hosts=xvr%3A1%2CH35%3A1%2Cxik%3A1%2Cudm%3A1%2Cots%3A1%2CH99%3A1%2Cyla%3A1%2Cixz%3A1%2Cziw%3A1%2CH253%3A1%2Cmwk%3A1%2Czci%3A1%2Cjjk%3A1%2Ceuw%3A1%2Cdwu%3A1%2Ceyl%3A1%2CH28%3A1%2Cbup%3A1%2Cdce%3A1%2CH30%3A1%2Coom%3A1%2Copx%3A1%2CH151%3A1%2Cpjw%3A1%2Cgzg%3A1%2Cywk%3A1%2Cdnm%3A1%2Cwjk%3A1%2Cuuk%3A1%2Cudt%3A1%2Czgf%3A1%2Cayv%3A1%2Crai%3A1%2Cktz%3A1%2Cdfh%3A1%2Clck%3A1%2CH117%3A1%2Chty%3A1%2Cszd%3A1%2Cbax%3A1%2Cymj%3A1%2Cjjg%3A1%2Chbz%3A1%2Cdui%3A1%2Cstj%3A1%2Cyqw%3A1%2Cddu%3A1%2Ccnt%3A1%2CH59%3A1%2Cyze%3A1%2CH80%3A1%2Ctif%3A1%2Cdvt%3A1%2Csjs%3A1%2Cviv%3A1%2Catx%3A1%2CH212%3A1%2Caiy%3A1%2Cqsc%3A1%2Cbro%3A1%2Capv%3A1%2Cvhh%3A1%2Cslt%3A1%2Cmlc%3A1%2Czsx%3A1%2CH155%3A1%2Cqih%3A1%2CH122%3A1%2CH32%3A1%2Cwjk%3A1%2Caso%3A1%2Cvpf%3A1%2Cbhq%3A1%2Cvrh%3A1%2CH37%3A1%2Cuuk%3A1%2Cwtu%3A1%2Chiz%3A1%2CH65%3A1%2CH68%3A1%2Czsx%3A1&legInt=&AwaitingReconsent=false",
                            }
                        )

                        response = await client.get(
                            url, headers=default_headers, cookies=default_cookies
                        )
                        if (
                            "content-encoding" in response.headers
                            and response.headers["content-encoding"] == "br"
                        ):
                            try:
                                response.decoded = brotli.decompress(response.content).decode(
                                    response.encoding
                                )
                            except Exception as e:
                                response.decoded = response.content.decode(response.encoding)
                        else:
                            response.decoded = response.content.decode(response.encoding)
                    responses[url] = response
                except Exception as e:
                    response = f"[ fetch_urls ] No response from url {url}: {e.__class__.__name__} :: {e}"
                    responses[url] = response

    url_iterator = iter(urls)
    token_send_channel, token_receive_channel = trio.open_memory_channel(0)

    async with trio.open_nursery() as nursery:
        async with token_receive_channel:
            nursery.start_soon(token_issuer, token_send_channel.clone(), len(urls))
            for _ in range(number_workers):
                nursery.start_soon(worker, url_iterator, token_receive_channel.clone())

    return responses


def fetch_all_responses(urls, workers=20):
    responses = trio.run(fetch_responses, urls, workers)
    return responses
