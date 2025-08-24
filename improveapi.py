# --- Utility: Extract tweet text from dict for frontend display ---
def extract_tweet_text(response):
    """
    Given a response like {'v2': "..."}, return only the tweet text.
    If response is not a dict, return as-is.
    """
    if isinstance(response, dict):
        return next(iter(response.values()), "")
    return str(response)


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any
import time
import logging
import os
import importlib
import requests

# --- Import your improve function
from improve import improve_tweet

# --- FastAPI app setup ---
app = FastAPI()

# --- CORS setup (add all domains/ports you need) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",         # local dev frontend
        "http://127.0.0.1:3000",         # alternative local
        "https://*.lovable.dev",         # Lovable preview domains
        "https://tweexter.co",           # main domain
        "https://www.tweexter.co",       # with www
        "https://api.tweexter.co",       # production API
        "https://api2.tweexter.co",      # new API2
        "https://preview--tweexter.lovable.app",  # Lovable preview
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Startup checks for environment, dependencies, and external services ---
@app.on_event("startup")
async def startup_checks():
    # 1. Check required environment variables
    required_env = ["SCORER_URL", "GEMINI_API_KEY"]
    missing = [var for var in required_env if not os.getenv(var)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    # 2. Check optional HuggingFace token
    if not os.getenv("HF_TOKEN"):
        logging.warning("HF_TOKEN not set. HuggingFace model downloads may fail if required.")

    # 3. Check dependencies
    try:
        importlib.import_module("google.generativeai")
    except ImportError:
        raise RuntimeError("google-generativeai is not installed. Run: pip install google-generativeai")
    try:
        importlib.import_module("requests")
    except ImportError:
        raise RuntimeError("requests is not installed. Run: pip install requests")

    # 4. Check external scorer service health
    scorer_url = os.getenv("SCORER_URL")
    healthz_url = scorer_url.rstrip("/") + "/healthz"
    try:
        resp = requests.get(healthz_url, timeout=5)
        if resp.status_code != 200:
            raise RuntimeError(f"SCORER_URL /healthz returned status {resp.status_code}")
    except Exception as e:
        raise RuntimeError(f"Could not reach SCORER_URL /healthz: {e}")

    logging.info("Startup checks passed: environment, dependencies, and external services are healthy.")

class ImproveRequest(BaseModel):
    text: str = Field(..., description="Original tweet text to improve", max_length=280)
    followers: int = Field(100, ge=1, description="Author follower count (default 100)")

    @validator("text")
    def non_empty_text(cls, v: str):
        if not v.strip():
            raise ValueError("text must not be empty")
        return v




class PredictedMetrics(BaseModel):
    likes: float
    retweets: float
    replies: float
    composite: float
    details: Optional[Dict[str, Any]] = None  # allow the nested dict

class ImproveResponse(BaseModel):
    improved_text: str
    predicted: Optional[PredictedMetrics] = None
    delta_vs_original: Optional[Dict[str, float]] = None
    details: Optional[Dict[str, Any]] = None

@app.post("/improve", response_model=ImproveResponse)
async def improve(req: ImproveRequest):
    start_time = time.perf_counter()
    try:
        result = improve_tweet(
            tweet_text=req.text,
            followers=req.followers,
            num_variants=20,
            return_all=False,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Improve failed: {e}")

    elapsed_ms = int((time.perf_counter() - start_time) * 1000)
    logging.info(f"Improve endpoint took {elapsed_ms} ms for text length {len(req.text)}")

    winner = result.get("winner")
    if not winner:
        return ImproveResponse(improved_text=req.text)

    improved_text = winner.get("text", req.text)
    # If improved_text is a dict like {'v2': '...'}, extract the value
    improved_text = extract_tweet_text(improved_text)
    predicted_dict = winner.get("predicted") or None
    delta_vs_original = winner.get("delta_vs_original")
    details = result.get("details")

    return ImproveResponse(
        improved_text=improved_text,
        predicted=PredictedMetrics(**predicted_dict) if predicted_dict else None,
        delta_vs_original=delta_vs_original,
        details=details,
    )

@app.get("/healthz")
def healthz():
    return {"ok": True}
