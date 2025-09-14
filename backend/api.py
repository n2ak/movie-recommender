import os
import sys
import time
import mlflow
import asyncio
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from movie_recommender.recommender import Recommender, Request as RecomRequest, Response as RecomResponse
sys.path.append("./dags")


TIMEOUT = 0.1
BATCH_SIZE = 64
queue: asyncio.Queue[tuple[RecomRequest, asyncio.Future]] = asyncio.Queue()


async def worker():
    while True:
        requests: list[tuple[RecomRequest, asyncio.Future]] = []

        while len(requests) < BATCH_SIZE:
            try:
                req, fut = await asyncio.wait_for(queue.get(), timeout=TIMEOUT)
                requests.append((req, fut))
            except asyncio.TimeoutError:
                break

        if requests:
            batch = [r for r, _ in requests]
            print(f"Processing {len(batch)}/{BATCH_SIZE} requests")
            start = time.time()
            try:
                preds = Recommender.batched(batch)
                for pred, (_, fut) in zip(preds, requests):
                    fut.set_result(pred)
            except Exception as e:
                print("Exception", e)
                for _, fut in requests:
                    if not fut.done():
                        fut.set_exception(e)
            process_time = (time.time() - start) * 1000
            print(f"Processing took {process_time:.0f}ms")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up...")
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    Recommender.load_all_models(exclude=["dlrm_cpu", "dlrm_cuda", "xgb_cuda"])
    asyncio.create_task(worker())
    print("\n\nServer is up...\n\n")
    yield
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)


# @app.middleware("http")
# async def log_requests(request: Request, call_next):
#     response = await call_next(request)
#     return response


@app.post("/movies-recom")
async def movies_recom(data: RecomRequest):
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    await queue.put((data, future))
    return RecomResponse(
        userId=data.userId,
        result=await future,
        error=None,
        status_code=200,
        time=0
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000, log_config=None)
