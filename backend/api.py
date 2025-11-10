import os
import sys
import argparse
import litserve

sys.path.append(os.path.abspath(__file__))


def load(api_name: str):
    match api_name:
        case "dlrm":
            from movie_recommender.dlrm.dlrm_api import DLRMLitAPI
            return DLRMLitAPI(api_path="/dlrm")
        case "xgb":
            raise Exception(f"XGB not available for now.")
        case "embedding":
            from movie_recommender.simsearch.embedding_api import EmbeddingAPI
            return EmbeddingAPI(max_batch_size=64, batch_timeout=1.0, api_path="/embedding")
        case _:
            raise Exception(f"Invalid model type: {api_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("api_names", type=str, help="The apis to serve.")
    args = parser.parse_args()

    api_names = args.api_names.split(",")
    apis = [load(api)for api in api_names]

    server = litserve.LitServer(apis)
    server.run(port=int(os.getenv("API_PORT", "8000")))
