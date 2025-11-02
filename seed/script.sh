#!/bin/bash
set -e

echo "Generating Prisma client..."
python -m prisma generate

echo "Pushing schema..."
python -m prisma db push


echo "Fetching data..."
python get_ds.py

echo "Seeding db..."
python main.py