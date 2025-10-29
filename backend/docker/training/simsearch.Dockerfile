FROM python:3.12-slim-bookworm

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY movie_recommender movie_recommender
COPY training training


CMD ["python","training/similarity_search/simsearch.py"]