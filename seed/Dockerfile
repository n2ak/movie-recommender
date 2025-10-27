FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && \
    apt-get install -y curl openssl && \
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

COPY seed/requirements.txt .
RUN pip install --upgrade --no-cache-dir -r requirements.txt

COPY seed/ ./
RUN npx prisma@5.17.0 generate

ENTRYPOINT ["bash", "-c", "python get_ds.py && python main.py"]
