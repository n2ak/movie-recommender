FROM apache/airflow:3.0.6
COPY requirements.txt /
RUN pip install --upgrade pip
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r /requirements.txt

RUN pip install --no-cache-dir lightning==2.5.0

COPY movie_recommender movie_recommender 
COPY setup.py requirements.txt .
RUN pip install -e .
