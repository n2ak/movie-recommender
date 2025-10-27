FROM node:20-slim AS builder

WORKDIR /app

RUN npm install -g prisma

COPY ../frontend/packages/database/prisma/schema.prisma prisma/schema.prisma

RUN npx prisma generate


FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y openssl curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY --from=builder /app/prisma /app/prisma

WORKDIR /app

ENTRYPOINT ["bash", "-c", "python get_ds.py | python main.py"]
