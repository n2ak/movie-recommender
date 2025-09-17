#!/bin/bash

set -e
echo "Destroying existing db..."
pushd frontend
    pnpm run db:reset --force
    pnpm run db:migrate
popd


echo "Downloading new dataset..."
paths=$(python backend/scripts/get_ds.py)

echo "Uploading new dataset..."
pushd frontend
    pnpm run db:seed $paths
popd
