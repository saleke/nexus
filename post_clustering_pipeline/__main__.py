"""Run the API with ``python -m post_clustering_pipeline``.

Structured JSON logging is configured up front (log.py). ``log_config=None``
keeps uvicorn from installing its own handlers over ours.
"""

import uvicorn

from .log import configure_json_logging


if __name__ == "__main__":
    configure_json_logging()
    uvicorn.run("post_clustering_pipeline.api:app", host="0.0.0.0", port=8000, log_config=None)