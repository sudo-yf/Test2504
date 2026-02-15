# Installation

## Prerequisites

- Python 3.10 or 3.11
- `uv` package manager
- Optional: Docker 24+ and Docker Compose

## Use UV (Recommended)

```bash
uv sync
cp .env.example .env
```

Install training and model extras:

```bash
uv sync --extra train --extra models --extra dev
```

Run:

```bash
uv run python main.py
```

## Use Docker

Build image:

```bash
docker compose build
```

Run application container:

```bash
docker compose run --rm emotisense
```

## Environment Variables

Configure `.env`:

- `DEEPSEEK_API_KEY`: optional, used for trend analysis API output.
