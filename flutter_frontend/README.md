# Gro AI Flutter Frontend

Flutter client for the Gro AI chat, inventory, and procurement workflows.

## Run Locally

From the repository root:

```bash
cd flutter_frontend
flutter pub get
flutter run -d chrome \
  --dart-define=API_BASE_URL=http://localhost:8000/api \
  --dart-define=WS_URL=ws://localhost:8000/ws
```

The `--dart-define` values are recommended for local review because the hosted
Cloud Run backend may be disabled outside demo windows to avoid cloud charges.

## Notes

- Auth tokens are stored with `flutter_secure_storage`.
- The app talks to the FastAPI backend over HTTP and WebSocket.
- The supported AI path is `@gro` commands backed by OpenAI/Gemini through the
  backend.
