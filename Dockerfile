FROM ghcr.io/astral-sh/uv:python3.12-bookworm

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-group train

COPY . .

CMD ["uv", "run", "python", "-c", "print('worker image ready')"]