import json
from bs4 import BeautifulSoup
from driver import Driver, get, parallel, read, append, movies_path, reviews_path, get_imdbid_from_url
import re


class ReviewScraper(Driver):
    def get_review_ids(self, nclicks):
        html = self.get_html(nclicks)
        soup = BeautifulSoup(html, "html.parser")
        links = soup.find_all("a", class_="ipc-title-link-wrapper")
        review_links = []
        for a in links:
            href = a.get("href")
            match = re.match(r"\/title\/.+\/review\/(.+)\/", href)
            if match is None:
                continue
            review_id = match.group(1)
            review_links.append(review_id)
        return review_links

    def get_review_data(self, review_id):
        url = f"https://www.imdb.com/title/tt0111161/review/{review_id}"
        print("Getting review for", url)
        try:
            html = get(url)
            if html is None:
                return None
            soup = BeautifulSoup(html, "html.parser")
            jsonData = soup.find('script', {"type": "application/json"})
            if jsonData is None:
                card = soup.find("div", class_="ipc-list-card__content")
                children = list(card.children)

                rating = children[0].find(
                    "span", class_="ipc-rating-star--rating").get_text(strip=True)
                review_title = children[1].find(
                    "h3", class_="ipc-title__text").get_text(strip=True)
                is_spoiler = children[2].name == "button"
                # TODO
                # review_body = list(children[2].children)[0].get_text(strip=True)
                # print("rating", rating)
                # print("review_title", review_title)
                # print("review_body", review_body)
            else:
                jsonData = json.loads(jsonData.string)
                review_data = jsonData["props"]["pageProps"]["reviewData"]
                data = dict(
                    id=review_data["id"],
                    author=review_data["author"]["nickName"],
                    title=review_data["summary"]["originalText"],
                    body=review_data["text"]["originalText"]["plaidHtml"],
                    rating=review_data["authorRating"],
                    date=review_data["submissionDate"],
                    likes=review_data["helpfulness"]["upVotes"],
                    dislikes=review_data["helpfulness"]["downVotes"],
                )
                return data
        except Exception as e:
            print(e)
            return None

    def run(self, imdbid, nclicks, filename):
        ids = self.get_review_ids(nclicks)
        old = read(filename)
        ids = [id for id in ids if f"{imdbid}_{id}" not in old]
        print(f"Getting data for {len(ids)} reviews")
        all_data = parallel(self.get_review_data, ids)
        new_data = {}
        for id, review in zip(ids, all_data):
            if review is not None:
                if imdbid not in new_data:
                    new_data[imdbid] = {}
                new_data[imdbid][id] = review
        append(filename, new_data)


def get_review_url(imdbid):
    return f"https://www.imdb.com/title/{imdbid}/reviews/"


def get_movie_ids(filename):
    movies = read(filename)
    ids = []
    for title, movie in movies.items():
        if movie is None:
            print(url)
            continue
        url = movie["url"]
        imdbid = get_imdbid_from_url(url)
        if imdbid is not None:
            ids.append(imdbid)
    return ids


if __name__ == "__main__":
    pass
    i = 0
    ids = get_movie_ids(movies_path)
    kk = read(reviews_path)
    print(len(ids), "ids")
    ids = [id for id in ids if id not in kk]
    print(len(ids), "ids")

    def run(imdbid):
        url = get_review_url(imdbid)
        scraper = ReviewScraper(url, "25 more")
        scraper.run(imdbid, 1, reviews_path)
    if False:
        parallel(run, ids, max_workers=2)
        pass
    else:
        for i, id in enumerate(ids):
            print(i+1)
            run(id)
