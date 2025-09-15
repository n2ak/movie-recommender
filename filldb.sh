#!/bin/bash

set -e
pushd frontend
    pnpm run db:reset
    pnpm run db:migrate
popd
pushd backend
    paths=$(python scripts/get_ds.py)
popd
pushd frontend
    pnpm run db:seed $paths
popd
