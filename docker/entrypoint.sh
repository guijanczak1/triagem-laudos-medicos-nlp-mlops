#!/bin/sh
# Entrypoint for docker/Dockerfile.api. Runs uvicorn against the ASGI app
# module-level instance exposed at src/triagem/serving/api.py (`app`, built
# by create_app() with the model loaded once in the lifespan startup hook).
#
# Honors the same TRIAGEM_ env-var convention as Settings (config.py):
# TRIAGEM_API_PORT overrides the bind port (default 8000, matching
# Settings.api_port and the image's EXPOSE 8000). UVICORN_WORKERS is an
# uvicorn-only knob (not a Settings field) for scaling worker processes.
set -eu

PORT="${TRIAGEM_API_PORT:-8000}"
WORKERS="${UVICORN_WORKERS:-1}"

exec uvicorn triagem.serving.api:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --workers "$WORKERS"
