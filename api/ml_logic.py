# api/ml_logic.py

# This file is now configured for PyTorch.
# Ensure your requirements.txt has 'torch' and 'transformers', and NOT 'tensorflow'.

import os
import logging
from transformers import pipeline
import yake

# --- Configuration ---
# 't5-small' is a great lightweight choice for memory-constrained environments.
SUMMARIZER_MODEL_NAME = "t5-small"

# YAKE! parameters for keyword extraction
YAKE_LANGUAGE = "en"
YAKE_MAX_NGRAM_SIZE = 3
YAKE_DEDUP_THRESHOLD = 0.9 # How similar keywords can be (lower means more deduplication)
YAKE_NUM_KEYWORDS = 10 # How many keywords to extract

# --- Global Variables for Loaded Models/Pipelines ---
# These will be loaded once by the load_models() function on startup.
# They are initialized to None.
summarization_pipeline_pt = None
kw_extractor = None

# Set up logging
logger = logging.getLogger(__name__)


def load_models() -> bool:
    """
    Loads the summarization pipeline (using PyTorch) and the keyword extractor.
    This function is intended to be called once on API startup via the lifespan manager.
    Returns:
        bool: True if models loaded successfully, False otherwise.
    """
    global summarization_pipeline_pt, kw_extractor # Declare that we are modifying the global variables

    # Optional: Check if already loaded to prevent reloading.
    if summarization_pipeline_pt is not None and kw_extractor is not None:
        logger.info("Models appear to be already loaded.")
        return True

    try:
        # --- Load Summarization Model (PyTorch) ---
        logger.info(f"Attempting to load PyTorch summarization pipeline: {SUMMARIZER_MODEL_NAME}...")
        summarization_pipeline_pt = pipeline(
            "summarization",
            model=SUMMARIZER_MODEL_NAME,
            tokenizer=SUMMARIZER_MODEL_NAME,
            framework="pt"  # <-- Key change: Use the PyTorch backend
        )
        logger.info(f"PyTorch summarization pipeline '{SUMMARIZER_MODEL_NAME}' loaded successfully.")

        # --- Initialize YAKE! Keyword Extractor ---
        logger.info("Initializing YAKE! keyword extractor...")
        kw_extractor = yake.KeywordExtractor(
            lan=YAKE_LANGUAGE,
            n=YAKE_MAX_NGRAM_SIZE,
            dedupLim=YAKE_DEDUP_THRESHOLD,
            top=YAKE_NUM_KEYWORDS,
            features=None
        )
        logger.info("YAKE! keyword extractor initialized successfully.")
        
        return True  # Indicate success

    except Exception as e:
        logger.error(f"Fatal error during model loading: {e}", exc_info=True)
        # Ensure partially loaded models are reset to None so are_models_ready() reflects the failure
        summarization_pipeline_pt = None
        kw_extractor = None
        return False  # Indicate failure

def are_models_ready() -> bool:
    """Checks if both essential models/pipelines are loaded and ready."""
    return summarization_pipeline_pt is not None and kw_extractor is not None

def get_summary(text: str, max_length: int = 150, min_length: int = 30) -> str:
    """Generates a summary for the given text using the loaded PyTorch pipeline."""
    if not are_models_ready():
        logger.warning("Summarization pipeline not ready. Cannot generate summary.")
        return "Error: Summarization model is not available at the moment."
    try:
        logger.info(f"Generating summary for text (approx length: {len(text.split())} words)...")
        result = summarization_pipeline_pt(text, max_length=max_length, min_length=min_length, do_sample=False)
        summary = result[0]['summary_text']
        logger.info(f"Summary generated (approx length: {len(summary.split())} words).")
        return summary
    except Exception as e:
        logger.error(f"Error during summarization task: {e}", exc_info=True)
        return f"Error: Could not generate summary due to an internal issue ({type(e).__name__})."

def get_tags(text: str) -> list[str]:
    """Extracts keywords/tags from the given text using YAKE!"""
    if not are_models_ready():
        logger.warning("Keyword extractor not ready. Cannot generate tags.")
        return ["Error: Tagging model is not available at the moment."]
    try:
        logger.info(f"Extracting tags for text (approx length: {len(text.split())} words)...")
        keywords_with_scores = kw_extractor.extract_keywords(text)
        tags = [kw[0] for kw in keywords_with_scores]
        logger.info(f"Tags extracted: {tags}")
        return tags
    except Exception as e:
        logger.error(f"Error during tag extraction task: {e}", exc_info=True)
        return [f"Error: Could not extract tags due to an internal issue ({type(e).__name__})."]