import os
import argparse
import litserve

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model", type=str, help="The model type to serve.", choices=["pytorch", "xgb"])
    args = parser.parse_args()

    match args.model:
        case "pytorch":
            from movie_recommender.dlrm.dlrm_api import DLRMLitAPI
            api = DLRMLitAPI()
        case "xgb":
            raise Exception(f"XGB not available for now.")
        case _:
            raise Exception(f"Invalid model type: {args.model}")

    server = litserve.LitServer(api)
    server.run(port=int(os.getenv("API_PORT", "8000")))
