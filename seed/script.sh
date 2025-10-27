#!/bin/bash
set -e

echo "Generating Prisma client..."
npx prisma@5.17.0 generate

echo "Applying migrations..."
# npx prisma@5.17.0 migrate deploy
npx prisma@5.17.0 migrate dev --name init

echo "Fetching data..."
python get_ds.py

echo "Starting main application..."
python main.py