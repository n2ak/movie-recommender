import os
from driver import Driver, html_path

BUTTON_TEXT = "50 more"
URL = "https://www.imdb.com/search/title/?title_type=tv_series,feature&sort=num_votes,desc"


def get_file_or_url(filename: str, url: str):
    filename = os.path.abspath(filename)
    if os.path.exists(filename):
        return f"file:///{filename.replace(os.sep, '/')}"
    return url


if __name__ == "__main__":
    nclicks = 200
    Driver(URL, BUTTON_TEXT).get_html(nclicks, html_path)
