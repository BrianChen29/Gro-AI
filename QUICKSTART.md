# Gro AI Quickstart

This guide runs the project locally. The previous GCP Cloud Run / Cloud SQL
deployment may be disabled to avoid cloud charges, so use localhost for review.

## Prerequisites

- Python 3.11+
- MySQL 8+
- Flutter SDK 3.x
- OpenAI API key or Gemini API key for AI features

## 1. Database

```bash
mysql -u root -p < sql/schema.sql
```

This creates a local development database named `groceryshopperai` and a
development user named `chatuser` with password `chatpass`.

## 2. Backend Environment

```bash
cp backend/.env.example backend/.env
```

Edit `backend/.env` and fill in a long local `JWT_SECRET` plus at least one LLM
API key:

```bash
OPENAI_API_KEY=...
GEMINI_API_KEY=...
```

## 3. Backend Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4. Load Grocery Catalog

```bash
cd backend
python load_groceries.py
cd ..
```

## 5. Run Backend

```bash
cd backend
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Backend URL:

```text
http://localhost:8000
```

## 6. Run Flutter

Open a second terminal:

```bash
cd flutter_frontend
flutter pub get
flutter run -d chrome \
  --dart-define=API_BASE_URL=http://localhost:8000/api \
  --dart-define=WS_URL=ws://localhost:8000/ws
```

## Demo Flow

1. Create an account.
2. Create a room.
3. Add inventory:

```text
@inventory
Tomatoes, 10, 20
Olive oil, 3, 5
Cheese, 12, 4
```

4. Try AI commands:

```text
@gro analyze
@gro menu
@gro restock
@gro plan
```

## Notes

- LLM features require valid OpenAI or Gemini credentials.
- Vector search uses an embedding SQLite cache. If the cache is unavailable,
  the app still supports auth, rooms, chat, inventory, and shopping lists.
- TinyLlama/Ollama was an earlier prototype path and is not part of the current
  supported local setup.
