import flask as F
import functools

app = F.Flask()

model = None


def parse_context(request):
    return None


def parse_response(data, resp):
    return None


@functools.lru_cache()
def run_model(input, range):
    return None


@app.get("/movies-recom/")
def get_movies_recom():
    data = parse_context(F.request)
    data = run_model(data)
    resp = parse_response(data, F.make_response())
    return resp


if __name__ == "__main__":
    host = "localhost"
    port = 3333

    app.run(host, port)
