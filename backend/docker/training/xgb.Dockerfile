# TODO use a different image
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir xgboost optuna

COPY movie_recommender movie_recommender
COPY training training


ENV CUDA_VISIBLE_DEVICES=0
CMD ["python","training/xgb.py"]