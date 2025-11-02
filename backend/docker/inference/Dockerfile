FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY docker/inference/torch_requirements.txt .
RUN pip install --no-cache-dir -r torch_requirements.txt
RUN pip install --no-cache-dir sentence_transformers

RUN python -c 'from sentence_transformers import SentenceTransformer;SentenceTransformer("all-MiniLM-L6-v2")'

COPY movie_recommender movie_recommender
# for the loading/pickle to work
COPY training training
COPY api.py .
COPY embedding_api.py .

ENV CUDA_VISIBLE_DEVICES=0

CMD ["python","api.py","embedding"]