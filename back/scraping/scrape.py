from driver import get, read, append, movie_links_path, movies_path
from concurrent.futures import ThreadPoolExecutor
import re
import json
from bs4 import BeautifulSoup, ResultSet


def get_keys(dict, keys):
    return {k: dict[k] for k in keys if k in dict}


class Scraper:
    def get_movie_data(self, link):
        try:
            print("Getting movie data of", link)
            html = get(link)
            if html is None:
                print("Error getting movie data", link)
                return
            soup = BeautifulSoup(html, "html.parser")
            jsonData = soup.find('script', {"type": "application/ld+json"})
            # title = soup.find('title')
            # ratingValue = soup.find("span", {"itemprop": "ratingValue"})
            a = json.loads(jsonData.string)
            data = get_keys(a, ["name", "url", "image",
                            "description", "aggregateRating", "genre", "datePublished", "keywords", "actor", "creator", ])
            return data
        except:
            return None

    def get_movie_links(self, url: str):
        movies = {}
        print("Getting movie links from", url)
        html = get(url)
        if html is None:
            return {}

        def get_link(a: ResultSet):
            href = a.get("href")
            h3 = a.find("h3", class_="ipc-title__text")
            h3 = h3.get_text(strip=True)
            title = re.match("(^[0-9]*\. )?(.+)$", h3)
            if title is None:
                print("Error matching:", h3)
            else:
                title = title.group(2)
                if title is not None:
                    movies[title] = f"https://www.imdb.com{href}"
                else:
                    print("No 2nd group:", h3)
        bs = BeautifulSoup(html, "html.parser")
        res = bs.find_all("a", class_='ipc-title-link-wrapper')
        print(len(res))
        for a in res:
            get_link(a)
        return movies


def get_movie_links(urls):
    scraper = Scraper()
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = executor.map(scraper.get_movie_links, urls)
    result = {}
    for res in futures:
        result.update(res)
    new_links = [link for link in result if link not in read(movie_links_path)]
    print("new links:", len(new_links))
    append(movie_links_path, result)


def get_movies_data():
    scraper = Scraper()
    links = read(movie_links_path)
    movies = read(movies_path)
    links = {title: link for title,
             link in links.items() if title not in movies}
    print(f"Getting data for {len(links)} movies.")
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = executor.map(scraper.get_movie_data, links.values())
    for title, result in zip(links.keys(), futures):
        if result is not None:
            movies[title] = result
    append(movies_path, movies)


if __name__ == "__main__":
    urls = {
        "https://www.imdb.com/chart/top/?ref_=nv_mv_250",
        "https://www.imdb.com/chart/moviemeter/?ref_=nv_mv_mpm",
        "https://www.imdb.com/chart/toptv/?ref_=chttentp_ql_6",
        "https://www.imdb.com/search/title/?title_type=tv_series,feature&sort=num_votes,desc",
        # f"file:///{filename}"
    }
    # if not os.path.exists(filename):
    #     Driver(URL).run(200, filename)

    get_movie_links(urls)
    get_movies_data()
