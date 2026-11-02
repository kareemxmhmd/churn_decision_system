# Churn Dashboard (`index.html`)

A single-file web page that acts as a simple front end for the churn prediction **FastAPI** service. It only sends requests to the API and shows the results. All the prediction logic lives in the backend.

**Endpoints used:** `GET /health`, `POST /predict`, `POST /predict/batch`

## Run it

1. Start the FastAPI backend (default: `http://127.0.0.1:8000`) with CORS enabled.
2. Open `index.html` in a browser.
3. If the backend runs somewhere else, set its URL in **Settings**.

## Note

This page was generated with AI (Claude) as a convenience tool for using the API.