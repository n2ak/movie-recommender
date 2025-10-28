# TODO use a different image
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY movie_recommender movie_recommender
COPY training/simsearch.py train_simsearch.py


ENV CUDA_VISIBLE_DEVICES=0