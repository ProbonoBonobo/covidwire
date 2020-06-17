from typing import List, Iterator
import trio
import httpx
from spider.common import default_headers


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

                print(f'[{round(trio.current_time(), 2)}] Start loading link: {url}')
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(url, headers=default_headers, cookies={"ccpa-state": "No"})
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


def fetch_all_responses(urls, workers=10):
    responses = trio.run(fetch_responses, urls, workers)
    return responses
