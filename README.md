### Seed

```bash
./filldb.sh
```

### build training docker images

```bash
cd backend
./build_training_docker_images.sh
```
### to run

```bash
docker compose --profile backend_api up -d
cd frontend && pnpm run dev

# OR

docker compose --profile web up -d
```
### to start airflow

```bash
docker compose --profile airflow up -d
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

