# Gro AI

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)
![Flutter](https://img.shields.io/badge/Flutter-Mobile%20%2B%20Web-02569B?logo=flutter&logoColor=white)
![MySQL](https://img.shields.io/badge/MySQL-Cloud%20SQL-4479A1?logo=mysql&logoColor=white)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-Async%20ORM-D71F00?logo=sqlalchemy&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-LLM-412991?logo=openai&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini-LLM-4285F4?logo=googlegemini&logoColor=white)
![WebSocket](https://img.shields.io/badge/WebSocket-Realtime%20Chat-4B5563)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)
![Cloud Run](https://img.shields.io/badge/GCP-Cloud%20Run-4285F4?logo=googlecloud&logoColor=white)
![RAG](https://img.shields.io/badge/RAG-Vector%20Search-7C3AED)

Gro AI is a team-built B2B grocery procurement platform that turns group chat,
restaurant inventory, and grocery catalog data into AI-assisted purchasing
workflows.

The app combines a Flutter client, FastAPI backend, MySQL persistence, real-time
WebSocket chat, and OpenAI/Gemini-powered planning modules for inventory
analysis, menu suggestions, restock planning, and procurement list generation.

## Project Context

Gro AI was developed as a USC Applied Data Science team project with a
product-oriented mindset. The team conducted domain research and spoke with
grocery managers and employees to understand procurement, inventory, and
communication pain points in real grocery operations.

The project was designed as a functional prototype for a B2B grocery procurement
workflow, with features shaped around stakeholder needs such as inventory
tracking, group coordination, restock planning, menu ideation, and procurement
list generation.

While the system is not currently operated as a live commercial SaaS product, it
was built to demonstrate how GenAI, backend systems, real-time communication,
and inventory-aware workflows could be integrated into a practical procurement
assistant.

## Technical Highlights

- **Project type:** Full-stack GenAI application with retrieval-augmented
  procurement workflows
- **Core stack:** FastAPI, SQLAlchemy, MySQL/Cloud SQL, Flutter, WebSocket,
  OpenAI, Gemini, Docker, GCP Cloud Run, Cloud Build
- **AI features:** LLM command router, structured JSON generation, grocery-item
  embedding search, inventory-aware recommendations, procurement plan generation
- **Backend features:** Authentication, room management, group chat, inventory
  tracking, shopping lists, WebSocket messaging, and AI-command workflows
- **Deployment:** Backend was deployed to GCP Cloud Run with Cloud SQL and
  Cloud Build for GitHub-triggered deployment. The live cloud resources may be
  disabled outside demos to avoid ongoing costs.

## What The App Does

Gro AI helps a restaurant or grocery-buying team coordinate procurement inside a
shared chat room.

Users can:

- Create accounts, log in, create rooms, invite members, and chat in real time.
- Track inventory items with stock and safety-stock thresholds.
- Ask the AI assistant to analyze low-stock items.
- Generate menu ideas from available inventory.
- Generate restock recommendations from low-stock inventory.
- Generate a consolidated procurement plan from group chat context.
- Save generated procurement items into shopping lists.

## My Contributions

This was a group project. My primary contributions focused on backend
development and AI engineering:

- Built most FastAPI backend routes for auth, rooms, chat, inventory, shopping
  lists, WebSocket messaging, and AI-command flows.
- Integrated OpenAI and Gemini APIs through a shared LLM wrapper.
- Built the LLM command routing for `@gro`, `@gro analyze`, `@gro menu`,
  `@gro restock`, `@gro plan`, and `@inventory`.
- Implemented grocery embeddings and vector-based catalog matching for
  retrieval-augmented recommendations.
- Built the AI modules for inventory analysis, menu generation, restock
  planning, and procurement planning.
- Dockerized and deployed the backend to GCP Cloud Run connected to Cloud SQL.
- Configured Cloud Build for GitHub-triggered deployment.

## Current Model Support

The current maintained path supports:

- OpenAI chat completions, default model configured by `OPENAI_MODEL`
- Google Gemini, model configured by `GEMINI_MODEL`

TinyLlama/Ollama was prototyped earlier in the project, but it was disabled in
the final maintained path because OpenAI and Gemini gave better response quality
and practical latency for this application. Some legacy docs or scripts may
still reference that experiment, but TinyLlama is not part of the current
supported setup.

## Architecture

```text
Flutter app
  |
  | HTTP + WebSocket
  v
FastAPI backend
  |
  | SQLAlchemy async ORM
  v
MySQL / Cloud SQL

FastAPI backend
  |
  | LLM calls
  v
OpenAI / Gemini

FastAPI backend
  |
  | embedding lookup
  v
SQLite embedding cache + grocery_items table
```

The retrieval layer uses a lightweight embedding cache for grocery catalog items.
Because the catalog is relatively stable and only updated occasionally, item
embeddings can be precomputed and refreshed when new products are added. This
keeps retrieval simple, fast, and cost-efficient while still supporting
inventory-aware recommendations.

## AI Workflow

1. A user sends a chat message such as `@gro analyze` or `@gro plan`.
2. The backend stores and broadcasts the original chat message.
3. The command router in `backend/app.py` detects the AI command.
4. Inventory, chat history, and relevant grocery catalog matches are loaded.
5. The selected LLM provider generates structured JSON output.
6. The backend persists the AI event and broadcasts it over WebSocket.
7. Flutter renders the AI result as a structured card in the chat UI.

## Repository Structure

```text
.
├── backend/
│   ├── app.py                    # FastAPI routes, WebSocket, AI command router
│   ├── auth.py                   # JWT auth and password hashing
│   ├── db.py                     # SQLAlchemy models and async DB session
│   ├── llm.py                    # OpenAI/Gemini LLM wrapper and embeddings
│   ├── llm_modules/              # Inventory/menu/restock/procurement modules
│   ├── vector/                   # Embedding cache and vector search helpers
│   ├── load_groceries.py         # Grocery CSV loader
│   ├── GroceryDataset.csv        # Small grocery catalog sample
│   └── .env.example              # Local backend environment template
├── flutter_frontend/
│   ├── lib/                      # Flutter app source
│   ├── web/                      # Flutter web entrypoint
│   ├── ios/                      # iOS project files
│   └── pubspec.yaml              # Flutter dependencies
├── sql/
│   ├── schema.sql                # Local MySQL schema setup
│   └── migration_add_deleted_at.sql
├── Dockerfile                    # Backend container for Cloud Run
├── cloudbuild.yaml               # Cloud Build deployment config
├── requirements.txt              # Backend Python dependencies
└── QUICKSTART.md                 # Local setup notes
```

## Local Setup

### Prerequisites

- Python 3.11+
- MySQL 8+
- Flutter SDK 3.x
- Optional: OpenAI API key and/or Gemini API key

### 1. Create The Local Database

```bash
mysql -u root -p < sql/schema.sql
```

The schema creates a local development database named `groceryshopperai` and a
development user named `chatuser`. The password in `sql/schema.sql` is a
local-only placeholder for reproducible setup. Do not use it for production.

### 2. Configure Backend Environment

```bash
cp backend/.env.example backend/.env
```

Then edit `backend/.env` and add at least one LLM API key.

```bash
DATABASE_URL=mysql+asyncmy://chatuser:chatpass@127.0.0.1:3306/groceryshopperai
JWT_SECRET=replace-with-a-long-random-dev-secret
OPENAI_API_KEY=
GEMINI_API_KEY=
```

### 3. Install Backend Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Load Grocery Catalog Data

```bash
cd backend
python load_groceries.py
cd ..
```

### 5. Start The Backend

```bash
cd backend
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

The backend API will be available at:

```text
http://localhost:8000
```

### 6. Run The Flutter App

In a second terminal:

```bash
cd flutter_frontend
flutter pub get
flutter run -d chrome \
  --dart-define=API_BASE_URL=http://localhost:8000/api \
  --dart-define=WS_URL=ws://localhost:8000/ws
```

Without `--dart-define`, the frontend falls back to the deployed backend URL
configured in `flutter_frontend/lib/services/api_client.dart`. That hosted
backend may be disabled outside demo windows to avoid cloud charges, so local
review should use the localhost values above.

## Example AI Commands

Inside a chat room:

```text
@inventory
Tomatoes, 10, 20
Olive oil, 3, 5
Cheese, 12, 4
```

```text
@gro analyze
@gro menu
@gro restock
@gro plan
@gro What should we buy for a small dinner service?
```

## API Surface

Main backend routes include:

- `POST /api/signup`
- `POST /api/login`
- `GET /api/rooms`
- `POST /api/rooms`
- `DELETE /api/rooms/{room_id}`
- `GET /api/rooms/{room_id}/members`
- `POST /api/rooms/{room_id}/invite`
- `GET /api/rooms/{room_id}/messages`
- `POST /api/rooms/{room_id}/messages`
- `GET /api/users/llm-model`
- `PUT /api/users/llm-model`
- `GET /api/inventory`
- `POST /api/inventory`
- `DELETE /api/inventory/{product_id}`
- `GET /api/shopping-lists`
- `POST /api/shopping-lists`
- `DELETE /api/shopping-lists/{list_id}`
- `POST /api/shopping-lists/{list_id}/check-item`
- `WS /ws?room_id={room_id}`

## Reproducibility Notes

- Local setup requires MySQL and a valid `DATABASE_URL`.
- LLM features require an OpenAI or Gemini API key.
- Vector search uses an embedding SQLite cache. In Cloud Run, the app attempts
  to download this cache from GCS. In local development, vector matching will
  fall back to an empty result set if no local `embeddings.sqlite` file exists.
- The backend can still run basic auth, rooms, chat, inventory, and shopping
  list flows without the embedding cache.

## Known Limitations

- This is a functional product prototype developed in an academic team setting, not a currently operated commercial SaaS product.
- The original Cloud Run / Cloud SQL deployment may be disabled outside demo windows to avoid ongoing cloud costs.
- WebSocket connections are room-scoped but not independently authenticated at connection time.
- CORS is permissive for local development and demo purposes.
- Automated test coverage is limited and should be expanded before production use.
- Some legacy documentation files may still reference earlier prototypes or experiments.

## Future Improvements

- Migrate the embedding retrieval layer to a managed vector database if the
  product catalog grows significantly or requires frequent real-time updates.
- Add Alembic migrations instead of schema-only SQL setup.
- Add CI checks for backend tests and Flutter analysis.
- Add structured Pydantic validation for LLM JSON outputs.
- Add a small seed/demo script for recruiter-friendly local demos.
- Add screenshots or a short demo GIF to the README.
- Move production secrets to Secret Manager and document the Cloud Run/Cloud SQL
  deployment setup without exposing environment-specific values.
- Revisit local model support only if latency and output quality become
  competitive with API-backed models.
