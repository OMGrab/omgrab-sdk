FROM python:3.11-slim

# Install system dependencies first (changes rarely, cached well)
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    netcat-openbsd \
    iproute2 \
    iputils-ping \
    libgpiod-dev \
    python3-libgpiod \
    wireless-tools \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
RUN mkdir -p src/omgrab && touch src/omgrab/__init__.py \
    && SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0 pip install --no-cache-dir . \
    && pip uninstall -y omgrab

COPY src/ ./src/
ARG SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0
RUN SETUPTOOLS_SCM_PRETEND_VERSION=${SETUPTOOLS_SCM_PRETEND_VERSION} pip install --no-cache-dir --no-deps .

COPY app/ ./app/

# ARGs that change frequently go AFTER expensive build steps to preserve cache.
ARG AGENT_VERSION=v0.0.1
ENV AGENT_VERSION=${AGENT_VERSION}

ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "-m", "app"]
