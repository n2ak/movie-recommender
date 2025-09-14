# Dataset

[movielens](https://grouplens.org/datasets/movielens/)

### Seed

```bash
./filldb.sh
```

### build docker

```bash
cd backend
./build_docker.sh
```
### to test backend api

```bash
cd backend
pytest -v -s
```
### to test backend api from frontend

```bash
cd backend
MLFLOW_TRACKING_URI=... python api.py
cd ../frontend
pnpm run test:api
```