# api/main.py
import sys
print(f"FULL MAIN.PY - PYTHON EXEC (UVICORN WORKER): {sys.executable}") # Keep for verification

from fastapi import FastAPI, HTTPException, __version__ as fastapi_version # <<<< CORRECTED LINE
from contextlib import asynccontextmanager
import logging

# Import your Pydantic models if they are used by FastAPI directly (e.g. for response_model in root)
# from pydantic import BaseModel # If you have root-level models

# Import your ml_logic and other necessary components
from ml_logic import load_models, get_summary, get_tags, are_models_ready
from pydantic import BaseModel # Move Pydantic models here if they are defined below


logger = logging.getLogger(__name__)
print(f"FULL MAIN.PY - FastAPI version from import: {fastapi_version}")
# ... rest of your file ...

# --- Pydantic Models ---
class TextInput(BaseModel):
    text: str
    max_summary_length: int = 150
    min_summary_length: int = 30

class SummaryResponse(BaseModel):
    original_text: str
    summary: str

class TagsResponse(BaseModel):
    original_text: str
    tags: list[str]

class AnalysisResponse(BaseModel):
    original_text: str
    summary: str
    tags: list[str]
# --- End Pydantic Models ---


@asynccontextmanager
async def lifespan(app_param: FastAPI): # Changed 'app' to 'app_param' to avoid shadowing global 'app'
    logger.info("Application startup: Loading ML models...")
    success = load_models()
    if not success:
        logger.warning("ML Models did not load successfully on startup!")
    # You could store 'success' on app_param.state if needed by endpoints before they call are_models_ready()
    # For example: app_param.state.initial_models_loaded_successfully = success
    yield
    logger.info("Application shutdown.")

print("Attempting FULL FastAPI app with lifespan and other params...")
try:
    app = FastAPI(
        title="Smart Content Analyzer API",
        description="API for text summarization and tag extraction.",
        version="0.1.0",
        lifespan=lifespan # Keyword argument
    )
    print("FULL FastAPI app created successfully.")
except TypeError as e_type:
    print(f"GOT TYPEERROR WITH FULL APP: {e_type}")
    raise # Re-raise to see the traceback if it happens
except Exception as e_other:
    print(f"GOT OTHER EXCEPTION WITH FULL APP: {e_other}")
    raise # Re-raise


# --- Endpoints ---
@app.get("/")
async def root():
    return {"message": "Welcome to the Smart Content Analyzer API!"}

@app.post("/summarize", response_model=SummaryResponse)
async def summarize_text_endpoint(payload: TextInput):
    if not are_models_ready():
        raise HTTPException(status_code=503, detail="ML models are not available. Please try again later.")
    logger.info(f"Received text for summarization (length: {len(payload.text)} chars)")
    summary = get_summary(payload.text, payload.max_summary_length, payload.min_summary_length)
    if "Error:" in summary:
        raise HTTPException(status_code=500, detail=summary)
    return SummaryResponse(original_text=payload.text, summary=summary)

@app.post("/extract_tags", response_model=TagsResponse)
async def extract_tags_endpoint(payload: TextInput):
    if not are_models_ready():
        raise HTTPException(status_code=503, detail="ML models are not available. Please try again later.")
    logger.info(f"Received text for tag extraction (length: {len(payload.text)} chars)")
    tags = get_tags(payload.text)
    if tags and isinstance(tags, list) and tags and "Error:" in tags[0]:
         raise HTTPException(status_code=500, detail=tags[0])
    return TagsResponse(original_text=payload.text, tags=tags)

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_text_endpoint(payload: TextInput):
    if not are_models_ready():
        raise HTTPException(status_code=503, detail="ML models are not available. Please try again later.")
    logger.info(f"Received text for full analysis (length: {len(payload.text)} chars)")
    summary = get_summary(payload.text, payload.max_summary_length, payload.min_summary_length)
    tags = get_tags(payload.text)
    error_details = []
    summary_failed = "Error:" in summary
    tags_failed = isinstance(tags, list) and tags and "Error:" in tags[0]
    if summary_failed: error_details.append(f"Summarization failed: {summary}")
    if tags_failed: error_details.append(f"Tagging failed: {tags[0]}")
    if error_details: raise HTTPException(status_code=500, detail="; ".join(error_details))
    return AnalysisResponse(original_text=payload.text, summary=summary, tags=tags)