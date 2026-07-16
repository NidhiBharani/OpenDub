.PHONY: setup dev server web build test e2e lint gpu-extras

# one-time: install server (core) + web deps
setup:
	cd server && uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e '.[dev]'
	cd web && npm install

# optional heavy OSS model deps (GPU recommended)
gpu-extras:
	cd server && uv pip install --python .venv/bin/python -e '.[asr,separation,tts]'

# run both dev servers (API :8000, UI :5173 with proxy)
dev:
	@trap 'kill 0' INT; \
	( cd server && .venv/bin/uvicorn app.main:app --reload --port 8000 ) & \
	( cd web && npm run dev ) & \
	wait

server:
	cd server && .venv/bin/uvicorn app.main:app --reload --port 8000

web:
	cd web && npm run dev

# production: build UI, then serve everything from FastAPI on :8000
build:
	cd web && npm run build

serve: build
	cd server && .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000

test:
	cd server && .venv/bin/python -m pytest tests/ -x -q
	cd web && npm run test

# real-browser end-to-end (headless Chromium; auto-starts both servers)
e2e:
	cd web && npm run e2e

lint:
	cd server && .venv/bin/python -m ruff check app tests
	cd web && npx tsc -b
