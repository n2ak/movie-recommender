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

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y openssl && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /app /app

ENTRYPOINT ["bash", "-c", "python get_ds.py && python main.py"]
