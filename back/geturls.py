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


def get(i, id):
    try:
        if i % 100 == 0:
            print(i)
        u = url(id)
        resp = requests.get(u, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
        }, timeout=10)
        if resp.status_code != 200:
            # print(resp.content.decode())
            return (id, -1)
        soup = BeautifulSoup(resp.content.decode(), "lxml")
        [div] = soup.find_all("div", {
            "class": "sc-491663c0-7 dUfBfF"
        })
        [img] = div.findAll("img")
        return id, img["src"]
    except Exception as e:
        return (id, -1)
        raise e
    return (id, -1)


url(114709)


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
    future = asyncio.ensure_future(get_work_done(df))
    print("Running")
    loop.run_until_complete(future)
    print("Done Running")
    return future


def save(urls, covers_path):
    urlss = np.array(urls)
    print(urlss.shape)
    href = pd.DataFrame({"imdbId": urlss[:, 0], "href": urlss[:, 1]})
    old_covers = pd.read_csv(covers_path)
    covers = pd.concat([old_covers, href]).reset_index(drop=True)
    covers.imdbId = covers.imdbId.astype(int)
    covers = covers.drop_duplicates("imdbId")
    covers.to_csv(covers_path, index=False)
    print("Done", covers.shape)


if __name__ == "__main__":
    covers_path = "covers.csv"

    df = pd.read_csv("./dataset/ml-32m/links.csv")
    df2 = pd.read_csv(covers_path)
    df = df.merge(df2, on="imdbId", how="outer")
    df = df.sort_values("movieId").iloc[:13629+1].reset_index(drop=True)
    df = df.loc[df.href.isna()].reset_index(drop=True)
    df = df.iloc[:10000]
    if (df.shape[0] == 0):
        print("Nothing to be retrieved")
    else:
        future = main(df)
        urls = future.result()
        print("Saving")
        save(urls, covers_path)
