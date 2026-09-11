"""Run the API with ``python -m post_clustering_pipeline``."""

import uvicorn


if __name__ == "__main__":
    uvicorn.run("post_clustering_pipeline.api:app", host="0.0.0.0", port=8000)
