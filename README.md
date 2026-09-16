# Gro AI — AI-Powered Restaurant Procurement Platform

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)
![Flutter](https://img.shields.io/badge/Flutter-Mobile%20%2B%20Web-02569B?logo=flutter&logoColor=white)
![MySQL](https://img.shields.io/badge/MySQL-Cloud%20SQL-4479A1?logo=mysql&logoColor=white)
![Cloud Run](https://img.shields.io/badge/GCP-Cloud%20Run-4285F4?logo=googlecloud&logoColor=white)

**From inventory and group conversations to coordinated procurement.**

Gro AI connects restaurants with grocery suppliers through shared chat,
inventory tracking, and AI-assisted planning. A Flutter client brings these
workflows together, while a FastAPI backend combines business data,
OpenAI/Gemini integration, and semantic matching against a supplier catalog.

Developed as a USC Applied Data Science team project, Gro AI was shaped by
conversations with grocery managers and employees about fragmented ordering,
inventory management, and supplier communication.

## Product Experience

| Workflow | What users can do |
|---|---|
| Collaborative workspace | Create rooms, invite members, and coordinate through real-time chat |
| Inventory tracking | Record stock and safety-stock levels to identify restocking needs |
| Inventory analysis | Use `@gro analyze` to review stock and receive recommendations |
| Menu planning | Use `@gro menu` to suggest dishes from available inventory |
| Restock planning | Use `@gro restock` to generate suggestions informed by low stock and catalog matches |
| Chat-based procurement | Use `@gro plan` to consolidate the conversation into a shopping plan and match items to catalog products |
| Shopping lists | Save procurement items and track their completion |

For example, a user can add inventory in a room:

```text
@inventory
Tomatoes, 10, 20
Olive oil, 3, 5
Cheese, 12, 4
```

Then send `@gro restock` for low-stock recommendations or `@gro plan` after
discussing procurement needs with the room. AI results appear as structured
cards in the Flutter interface.

## My Contributions

I led backend development and LLM integration for the application-specific AI
workflows, collaborating with teammates on the database and Flutter interfaces.

- **AI workflows:** Implemented inventory analysis, menu generation, restocking,
  and chat-context procurement planning with OpenAI and Gemini.
- **Frontend integration:** Defined the AI event contract, processed structured
  model outputs, and enriched results with product data for Flutter cards.
- **Catalog grounding:** Built embedding-based product matching, SQLite vector
  caching, and retrieval of current product metadata from MySQL.
- **Retrieval extension:** Added a shared vector-store interface, an optional
  async Pinecone adapter, catalog synchronization commands, evaluation tooling,
  and batched search and product hydration.
- **Application integration:** Connected these workflows to the team's chat,
  room, inventory, shopping-list, and WebSocket services.
- **Cloud delivery:** Owned the FastAPI application's Docker/Cloud Build/
  Cloud Run deployment with Cloud SQL and GCS, and built and deployed the
  team-developed Flutter Web frontend to Firebase Hosting.

## Engineering Highlights

- **Interchangeable retrieval backends:** SQLite-cached vectors with NumPy
  exact cosine search by default, or an optional async Pinecone adapter.
- **Batched catalog matching:** Embed multiple queries in one provider request,
  bound concurrent vector searches, and load matched product rows in one
  catalog SQL query while preserving each query's ranking.
- **Explicit catalog maintenance:** Dry-run-first commands for initial vector
  synchronization and targeted product upserts/deletions.
- **Evaluation tooling:** Compute Hit Rate@K, Precision@K, Recall@K, and MRR@K
  from relevance labels; calibrate a score threshold and evaluate it on held-out
  cases before enabling it.
- **Separation of data responsibilities:** MySQL owns product details such as
  price and rating; the vector layer returns candidate product IDs and scores.
- **Testable integration:** Offline tests cover vector-store behavior,
  synchronization, evaluation, batching, and the catalog-grounding pipeline.

## Architecture

```text
Flutter client
    |
    | HTTP + WebSocket
    v
FastAPI — authentication, chat, inventory, shopping lists, AI routing
    |
    +-- SQLAlchemy --> MySQL / Cloud SQL
    |
    +-- AI workflows --> OpenAI / Gemini --> structured AI events
    |
    +-- Catalog retrieval
            |
            +-- OpenAI query embeddings
            +-- VectorStore: NumPy exact search / Pinecone adapter
            +-- Ranked product IDs --> MySQL product details
```

The structured AI commands use catalog matching in two ways:

- **Analyze / menu / restock:** Retrieve relevant products before generation
  and supply them alongside inventory context.
- **Plan:** Generate a procurement list from chat history, then match its item
  names to real catalog products.

See the [catalog retrieval guide](backend/vector/README.md) for backend
configuration, synchronization, and evaluation commands.

## Run Locally

Requirements: Python 3.11+, MySQL 8+, and Flutter SDK 3.x. AI chat requires a
configured OpenAI or Gemini provider; catalog embeddings require an OpenAI API
key regardless of the chat provider.

### 1. Install and configure

From the repository root in a fresh checkout:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp backend/.env.example backend/.env
mysql -u root -p < sql/schema.sql
```

Edit `backend/.env` with your database connection, a new `JWT_SECRET`, and
provider keys. The schema's development credentials are local-only examples.
Keep secrets out of version control.

### 2. Prepare the catalog

For the default memory backend:

```bash
cd backend
python load_groceries.py
python -m vector.embedding_loader
cd ..
```

The embedding step creates the local SQLite vector cache and calls the OpenAI
API, which may incur usage charges. If you already have a compatible cache,
configure `EMBEDDINGS_DB_PATH` instead of generating a new one. Pinecone setup
is covered in the [retrieval guide](backend/vector/README.md).

### 3. Start the backend

From the repository root:

```bash
cd backend
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

The local API documentation is available at
[localhost:8000/docs](http://localhost:8000/docs).

### 4. Start Flutter Web

In a second terminal, from the repository root:

```bash
cd flutter_frontend
flutter pub get
flutter run -d chrome \
  --dart-define=API_BASE_URL=http://localhost:8000/api \
  --dart-define=WS_URL=ws://localhost:8000/ws
```

Register an account, create a room, add inventory, and try the commands above.

## Tests

With backend dependencies installed, run the offline suite from the repository
root:

```bash
python -m pip install pytest
cd backend
python -m pytest tests -q
```

The suite uses controlled external-service boundaries and runs without live API
keys or database servers. It includes a cross-layer test of catalog retrieval,
similarity filtering, ranked product hydration, and context formatting.

## Deployment

The course application was deployed with **Cloud Run, Cloud SQL, GCS, and
Firebase Hosting**, using Docker and Cloud Build for the backend delivery path.
Deployment configuration is available in [Dockerfile](Dockerfile) and
[cloudbuild.yaml](cloudbuild.yaml).

Hosted demo resources may be paused between demonstrations; use the local
setup above to explore the project. These instructions target local/demo use;
review authentication, authorization, and deployment settings before exposing
an instance publicly.

## Code Guide

| Area | Entry point |
|---|---|
| API routes, WebSocket, and AI command routing | [backend/app.py](backend/app.py) |
| Database models and sessions | [backend/db.py](backend/db.py) |
| Chat provider integration | [backend/llm.py](backend/llm.py) |
| Application-specific AI workflows | [backend/llm_modules/](backend/llm_modules/) |
| Vector stores, synchronization, and evaluation | [backend/vector/](backend/vector/) |
| Offline backend tests | [backend/tests/](backend/tests/) |
| Flutter application | [flutter_frontend/lib/](flutter_frontend/lib/) |
