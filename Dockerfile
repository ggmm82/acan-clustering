# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm@sha256:e697c81897302a4c4333bade000d98038fbb21eb064f8ee1fd42d02ebcd9d337
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/matplotlib PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY requirements/docker.lock /app/requirements/docker.lock
# The optional secret supplies the session CA in managed build environments.
# No session trust material or proxy credential is retained in the image.
RUN --mount=type=secret,id=proxy_ca \
    if [ -f /run/secrets/proxy_ca ]; then export PIP_CERT=/run/secrets/proxy_ca; fi; \
    python -m pip install --no-cache-dir --require-hashes -r requirements/docker.lock
COPY pyproject.toml README.md LICENSE ./
COPY acan ./acan
RUN python -m pip install --no-cache-dir --no-deps --no-build-isolation . \
    && rm -rf /app/acan /app/build /app/*.egg-info \
    && useradd --uid 10001 --create-home acan
USER 10001
EXPOSE 4000
HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:4000/healthz', timeout=3)"
# Jobs are in process: keep exactly one worker, with threads for UI requests.
CMD ["gunicorn", "--bind=0.0.0.0:4000", "--workers=1", "--threads=8", "--timeout=120", "--control-socket=/tmp/gunicorn.ctl", "acan.vis.wsgi:application"]
