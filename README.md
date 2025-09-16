### Seed

```bash
docker compose db up -d  && ./filldb.sh
```

### build training docker images

```bash
cd backend && ./build_training_docker_images.sh
```
### to run

```bash
docker compose backend up -d
cd frontend && pnpm run dev

# OR

docker compose webapp up -d
```
### to start airflow

```bash
docker compose airflow up -d
```

### to test backend api

```bash
cd backend
MLFLOW_TRACKING_URI=... python -m pytest -v -s
```
### to test backend api from frontend

```bash
cd backend
MLFLOW_TRACKING_URI=... python api.py
cd ../frontend
pnpm run test:api
```