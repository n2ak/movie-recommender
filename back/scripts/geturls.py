from bs4 import BeautifulSoup
import requests
import asyncio
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np


def get_imdbid(row):
    return int(row.imdbId)


def url(id):
    return f"https://www.imdb.com/title/tt{id:07d}/"


possible_classes = reversed(["sc-491663c0-7 dUfBfF", "sc-491663c0-7 jmhiib"])


def get(i, id):
    err = -1
    try:
        if i % 100 == 0:
            print(i)
        u = url(id)
        resp = requests.get(u, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
        }, timeout=10)
        if resp.status_code != 200:
            # print(resp.content.decode())
            return (id, err)
        soup = BeautifulSoup(resp.content.decode(), "html.parser")
        err = -2
        div = None
        for c in possible_classes:
            divs = soup.find_all("div", {
                "class": c
            })
            if len(divs) == 0:
                continue
            div = divs[0]
            break
        if div is None:
            raise ""
        err = -3
        [img] = div.findAll("img")
        err = -4
        return id, img["src"]
    except Exception as e:
        pass
    return (id, err)


async def get_work_done(df):
    print("Getting", df.shape[0], "urls")
    print("Getting id:", df.imdbId[0], "first")
    with ThreadPoolExecutor(max_workers=10) as executor:
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(
                executor,
                get,
                # Allows us to pass in arguments to `lyrics_from_year`
                *(i, get_imdbid(row),)
            )
            for i, row in df.iterrows()
        ]
    return await asyncio.gather(*tasks)


def main(df):
    loop = asyncio.get_event_loop()
    tasks = get_work_done(df)

    print("Running")
    try:
        future = asyncio.ensure_future(tasks)
        loop.run_until_complete(future)
    except KeyboardInterrupt as e:
        print("Caught keyboard interrupt. Canceling tasks...")
        tasks.cancel()
        future.cancel()
        loop.run_forever()
        future.exception()
        tasks.exception()
    finally:
        loop.close()

    print("Done Running")
    return future


def save(urls, covers_path):
    urlss = np.array(urls)
    print(urlss.shape)
    href = pd.DataFrame({"imdbId": urlss[:, 0], "href": urlss[:, 1]})
    old_covers = read_csv(covers_path)
    covers = pd.concat([old_covers, href]).reset_index(drop=True)
    covers.imdbId = covers.imdbId.astype(int)
    covers = covers.drop_duplicates("imdbId", keep="last")
    covers.to_csv(covers_path, index=False)
    print("Done", covers.shape)


def load_df(covers_path, s=10000):
    df = read_csv("./dataset/ml-32m/links.csv")
    df2 = read_csv(covers_path)
    df = df.merge(df2, on="imdbId", how="outer")
    df = df.sort_values("movieId").iloc[:13629+1].reset_index(drop=True)
    df = df.loc[df.href.isin(["-1", "-2", "-3"])].reset_index(drop=True)
    print("Need", df.shape[0], "urls")
    df = df.iloc[:s]
    return df


if __name__ == "__main__":
    covers_path = "covers.csv"
    df = load_df(covers_path, s=4000)
    if (df.shape[0] == 0):
        print("Nothing to be retrieved")
    else:
        future = main(df)
        urls = future.result()
        print("Saving")
        save(urls, covers_path)
