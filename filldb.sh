#!/bin/bash

set -e
pushd frontend
    pnpm run db:migrate
popd
pushd back
    paths=$(python scripts/get_ds.py)
popd
pushd frontend
    pnpm run db:seed $paths
popd
