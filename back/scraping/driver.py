
import re
import json
from concurrent.futures import ThreadPoolExecutor
import requests
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time


class Driver:

    def __init__(self, url, BUTTON_TEXT):
        options = Options()
        options.add_argument("--headless")
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36")
        self.driver = webdriver.Chrome(options=options)
        print("GET", url)
        self.driver.get(url)
        time.sleep(3)
        self.BUTTON_TEXT = BUTTON_TEXT

    def get_html(self, nclicks, filename=None):
        try:
            print(f"Clicking {nclicks} times")
            for _ in range(nclicks):
                if self.click_more():
                    print("Loaded", self.BUTTON_TEXT)
                else:
                    print("Button not found or error")
                    break
        except KeyboardInterrupt:
            print("Loading stopped")
        html = self.driver.page_source
        if filename is not None:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(html)
                print("Saved to", filename)
        self.driver.quit()
        return html

    def get_button(self, wait=5):
        button = WebDriverWait(self.driver, wait).until(
            EC.element_to_be_clickable(
                (By.XPATH, f"//span[text()='{self.BUTTON_TEXT}']/ancestor::button"))
        )
        time.sleep(0.5)
        return button

    def click_more(self, sleep=3):
        try:
            button = self.get_button()
            self.driver.execute_script("arguments[0].click();", button)
            time.sleep(sleep)
            return True
        except Exception as e:
            print("Couldn't click button")
            return False


user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"


def get(url: str):
    if url.startswith("file:///"):
        with open(url.removeprefix("file:///"), encoding="utf-8") as f:
            html = f.read()
    else:
        resp = requests.get(url, headers={
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https: // www.imdb.com/chart/top-english-movies /?ref_ = chtbo_ql_4"
        })
        if resp.status_code != 200:
            print(f"Error {resp.status_code} getting url {url}")
            return None
        html = resp.text
    return html


def parallel(func, iter: list, max_workers=5):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = executor.map(func, iter)
    return list(futures)


def append(file, obj):
    data = read(file)
    data.update(obj)
    with open(file, "w") as f:
        f.write(json.dumps(data))


def read(file):
    import os
    if not os.path.exists(file):
        return {}
    with open(file, "r") as f:
        return json.loads(f.read())


def get_imdbid_from_url(url):
    match = re.match(".*\/title\/(.+)\/.*", url)
    if match is None:
        return None
    return match.group(1)


movies_path = "./data/data.json"
reviews_path = "./data/reviews.json"
movie_links_path = "./data/movie_links.json"
html_path = "./data/html.html"
