#!/bin/bash
set -e

echo "Generating Prisma client..."
npx prisma@5.17.0 generate

echo "Pushing schema..."
npx prisma@5.17.0 db push


echo "Fetching data..."
python get_ds.py

echo "Starting main application..."
python main.py