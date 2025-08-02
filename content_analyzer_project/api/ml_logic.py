import os
print("[ML_LOGIC DEBUG] Attempting to set CUDA_VISIBLE_DEVICES=-1 to force CPU")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress TF INFO/WARNING

# Try to make TF explicitly acknowledge no GPUs AFTER the env var is set
# This needs to happen before other TF components are heavily initialized by transformers
try:
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("[ML_LOGIC DEBUG] TensorFlow correctly sees NO GPUs. CPU mode should be active.")
    else:
        print(f"[ML_LOGIC DEBUG] WARNING: TensorFlow still sees GPUs: {gpus}")
except Exception as e_tf_config:
    print(f"[ML_LOGIC DEBUG] Error during tf.config for GPU visibility in ml_logic: {e_tf_config}")

# Now the rest of your imports
# import tensorflow as tf # Already imported above for the config
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM, pipeline
import yake
import logging
logger = logging.getLogger(__name__)
# ... rest of ml_logic.py ...
# --- Configuration ---
SUMMARIZER_MODEL_NAME = "t5-small" # Make sure this is the model you intend to use

# YAKE! parameters
YAKE_LANGUAGE = "en"
YAKE_MAX_NGRAM_SIZE = 3
YAKE_DEDUP_THRESHOLD = 0.9 # How similar keywords can be (lower means more deduplication)
YAKE_NUM_KEYWORDS = 10 # How many keywords to extract

# --- Global Variables for Loaded Models/Pipelines ---
# These will be loaded once by the load_models() function
summarization_pipeline_tf = None
kw_extractor = None

def load_models() -> bool:
    """
    Loads the summarization pipeline and keyword extractor.
    This function is intended to be called once on API startup.
    Returns:
        bool: True if models loaded successfully, False otherwise.
    """
    global summarization_pipeline_tf, kw_extractor # Declare we are modifying the global variables

    # Check if already loaded (optional, but good for idempotency)
    if summarization_pipeline_tf is not None and kw_extractor is not None:
        logger.info("Models appear to be already loaded (or an attempt was made).")
        return True # Or return based on actual success of previous load

    try:
        logger.info(f"Attempting to load summarization pipeline: {SUMMARIZER_MODEL_NAME}...")
        # Specify framework="tf" for TensorFlow pipeline
        summarization_pipeline_tf = pipeline(
            "summarization",
            model=SUMMARIZER_MODEL_NAME,
            tokenizer=SUMMARIZER_MODEL_NAME,
            framework="tf"
            # You might need to specify device if TF isn't picking GPU automatically,
            # but usually TF manages this if Metal plugin is working.
            # device=0 # for GPU 0, or -1 for CPU (though TF handles this differently)
        )
        logger.info(f"Summarization pipeline '{SUMMARIZER_MODEL_NAME}' loaded successfully.")

        logger.info("Initializing YAKE! keyword extractor...")
        kw_extractor = yake.KeywordExtractor(
            lan=YAKE_LANGUAGE,
            n=YAKE_MAX_NGRAM_SIZE,
            dedupLim=YAKE_DEDUP_THRESHOLD,
            top=YAKE_NUM_KEYWORDS,
            features=None # Use default features for YAKE!
        )
        logger.info("YAKE! keyword extractor initialized successfully.")
        return True # Indicate success

    except Exception as e:
        logger.error(f"Error loading one or more ML models: {e}", exc_info=True) # Log the full traceback
        # Ensure partially loaded models are None so are_models_ready() reflects failure
        summarization_pipeline_tf = None
        kw_extractor = None
        return False # Indicate failure

def are_models_ready() -> bool:
    """Checks if both essential models/pipelines are loaded."""
    return summarization_pipeline_tf is not None and kw_extractor is not None

def get_summary(text: str, max_length: int = 150, min_length: int = 30) -> str:
    """Generates a summary for the given text."""
    if not are_models_ready() or summarization_pipeline_tf is None: # Redundant check but safe
        logger.warning("Summarization pipeline not loaded. Cannot generate summary.")
        return "Error: Summarization model is not available at the moment."
    try:
        logger.info(f"Generating summary for text (approx length: {len(text.split())} words)...")
        # The pipeline expects a list of texts, even if it's just one.
        result = summarization_pipeline_tf(text, max_length=max_length, min_length=min_length, do_sample=False)
        summary = result[0]['summary_text']
        logger.info(f"Summary generated (approx length: {len(summary.split())} words).")
        return summary
    except Exception as e:
        logger.error(f"Error during summarization task: {e}", exc_info=True)
        return f"Error: Could not generate summary due to an internal issue ({type(e).__name__})."

def get_tags(text: str) -> list[str]: # Explicitly type hint list of strings
    """Extracts keywords/tags from the given text."""
    if not are_models_ready() or kw_extractor is None: # Redundant check but safe
        logger.warning("Keyword extractor not loaded. Cannot generate tags.")
        return ["Error: Tagging model is not available at the moment."]
    try:
        logger.info(f"Extracting tags for text (approx length: {len(text.split())} words)...")
        keywords_with_scores = kw_extractor.extract_keywords(text)
        # keywords_with_scores is a list of tuples (keyword_string, score)
        tags = [kw[0] for kw in keywords_with_scores]
        logger.info(f"Tags extracted: {tags}")
        return tags
    except Exception as e:
        logger.error(f"Error during tag extraction task: {e}", exc_info=True)
        return [f"Error: Could not extract tags due to an internal issue ({type(e).__name__})."]

# No direct call to load_models() here.
# It will be called by the lifespan manager in main.py when the FastAPI app starts.