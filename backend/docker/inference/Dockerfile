FROM python:3.12

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY docker/inference/torch_requirements.txt .
RUN pip install --no-cache-dir -r torch_requirements.txt
RUN pip install --no-cache-dir sentence_transformers

RUN python -c 'from sentence_transformers import SentenceTransformer;SentenceTransformer("all-MiniLM-L6-v2")'

RUN pip install --no-cache-dir boto3

COPY movie_recommender movie_recommender
COPY training training
COPY api.py .

# Remove CUDA_VISIBLE_DEVICES for CPU
CMD ["python","api.py","dlrm,embedding"]
