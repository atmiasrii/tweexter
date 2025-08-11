# api.py
# FastAPI wrapper for Tweexter predictions with follower-aware scaling.

from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Form
from pydantic import BaseModel, Field, validator
from starlette.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

# Import the base blended predictor
from final_prediction import predict_blended

# Import follower-scaling helpers from your final.py
from final import (
    load_cfg,
    apply_follower_scaling,
    pick_blend_weights,
    factor_for,
    baselines_for,
    baseline_weight,
)

app = FastAPI(title="Tweexter API", version="1.0.0")

# CORS (tweak origins as you like)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your frontend origin(s) in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Schemas ----------

class PredictRequest(BaseModel):
    text: str = Field(..., description="Tweet text to predict on", max_length=2000)
    followers: int = Field(..., ge=1, description="Author follower count")
    return_details: bool = Field(False, description="Include full breakdown")

    @validator("text")
    def non_empty_text(cls, v: str):
        if not v.strip():
            raise ValueError("text must not be empty")
        return v


class PredictResponse(BaseModel):
    likes: int
    retweets: int
    replies: int
    details: Optional[Dict[str, Any]] = None


# ---- shared compute so both JSON + FORM routes can call the same code ----
def _compute(text: str, followers: int, return_details: bool) -> PredictResponse:
    override_w = pick_blend_weights(followers)
    base = predict_blended(text, override_w)
    blended = base.get("blended", {"likes": 0.0, "retweets": 0.0, "replies": 0.0})

    cfg = load_cfg()
    adjusted = apply_follower_scaling(blended, followers, cfg)

    out_likes    = int(round(adjusted.get("likes", 0.0)))
    out_retweets = int(round(adjusted.get("retweets", 0.0)))
    out_replies  = int(round(adjusted.get("replies", 0.0)))

    payload = None
    if return_details:
        scales = {m: factor_for(m, followers, cfg) for m in ("likes", "retweets", "replies")}
        payload = {
            "followers": followers,
            "weights_used": base.get("weights_used", {}),
            "models_used": base.get("models_used", {}),
            "ml_raw": base.get("ml", {}),
            "persona_raw": base.get("persona", {}),
            "blended_before_scaling": blended,
            "scaling_factors": scales,
            "follower_baselines": baselines_for(followers, cfg),
            "baseline_weight": baseline_weight(followers, cfg),
            "config_snapshot": cfg,
        }

    return PredictResponse(likes=out_likes, retweets=out_retweets, replies=out_replies, details=payload)


# ---------- Routes ----------

@app.get("/healthz")
def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.get("/")
def root() -> Dict[str, str]:
    return {"name": "Tweexter API", "version": app.version}

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    """
    Predict engagement for a single tweet.
    """
    try:
        return await run_in_threadpool(lambda: _compute(req.text, req.followers, req.return_details))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}") from e


@app.post("/predict-form", response_model=PredictResponse)
async def predict_form(
    text: str = Form(..., description="Paste tweet text (multiline OK)"),
    followers: int = Form(..., ge=1, description="Author follower count"),
    return_details: bool = Form(False, description="Include full breakdown"),
) -> PredictResponse:
    try:
        return await run_in_threadpool(lambda: _compute(text, followers, return_details))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}") from e
