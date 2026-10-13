"""FastAPI instant-triage service. P95 target <100ms (no LLM in path)."""
from __future__ import annotations
import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.predict import TriageSystem
from src.escalation_rules import RULES_CARD

MODEL_DIR = os.environ.get("MODEL_DIR", "models")
system: TriageSystem | None = None

app = FastAPI(title="LifeFirstAI — OPD Triage Assistant", version="1.0.0")

class Intake(BaseModel):
    age: float | None = Field(default=None, examples=[45])
    sex: str = Field(default="M", examples=["F"])
    sbp: float | None = None
    dbp: float | None = None
    pulse: float | None = None
    temp_c: float | None = Field(default=None, description="Celsius; Fahrenheit auto-converted (e.g. 98.6)")
    spo2: float | None = None
    complaint: str = Field(default="", examples=["seene me dard with pasina"])
    conditions: str = ""
    medications: str = ""

@app.on_event("startup")
def _load():
    global system
    system = TriageSystem(MODEL_DIR)

@app.get("/health")
def health():
    return {"ok": True, "model_loaded": system is not None}

@app.get("/rules")
def rules():
    return {"escalation_rules": [{"id": i, "rule": r, "trigger": t} for i, r, t in RULES_CARD]}

@app.post("/triage")
def triage(x: Intake):
    assert system is not None, "model not loaded"
    o = system.predict_one(x.age, x.sex, x.sbp, x.dbp, x.pulse, x.temp_c, x.spo2,
                           x.complaint, x.conditions, x.medications)
    return JSONResponse(o)

@app.get("/", response_class=HTMLResponse)
def ui():
    p = os.path.join(os.path.dirname(__file__), "static", "index.html")
    return open(p, encoding="utf-8").read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
