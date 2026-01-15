# ST-GNN React Admin Dashboard

A modern React admin dashboard (Material UI) that:
- Logs in with static credentials from `.env` (demo only)
- Uploads Excel to FastAPI (`/upload_excel`)
- Trains model (`/process_train`)
- Predicts hotspots and renders markers on Google Maps (`/predict_hotspots`)

## Setup

```bash
cp .env.example .env
npm install
npm start
```

Open: http://localhost:3000

## Backend CORS (required)

Add this to your FastAPI app:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Security note
Frontend `.env` values are bundled into the app. This static login is for demos only.
