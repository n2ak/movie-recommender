#!/bin/bash

set -e

docker build -t pytorch_train -f training/Dockerfile_pytorch .
docker build -t xgb_train -f training/Dockerfile_xgb .

pushd workflow
    docker compose build
popd