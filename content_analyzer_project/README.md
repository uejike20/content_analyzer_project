# Smart Content Analyzer & Tagger

This project demonstrates a full-stack application that provides text summarization and keyword/tag extraction using AI models. It features:

*   **AI/ML Core:**
    *   Text Summarization: Using a pre-trained T5 model (`t5-small`) via Hugging Face Transformers with a TensorFlow backend.
    *   Keyword/Tag Extraction: Using the YAKE! (Yet Another Keyword Extractor) library.
*   **Backend API:** A FastAPI application exposing endpoints for summarization, tagging, and combined analysis. This API is Dockerized.
*   **Frontend UI:** An interactive web application built with Streamlit that consumes the FastAPI backend.

This project showcases skills in NLP, model deployment via API, and building simple user interfaces for AI applications, aligning with concepts of intelligent automation and data-driven insights.

## Project Structure

## Features

*   Input long text through a user-friendly web interface.
*   Get an AI-generated summary of the text.
*   Receive a list of relevant keywords/tags extracted from the text.
*   Control desired summary length via UI controls.

## Technologies Used

*   **Python 3.10** (or your specific Python version)
*   **AI/ML:**
    *   TensorFlow v2.16.2 
    *   TensorFlow-Metal v1.1.0 
    *   Hugging Face Transformers
    *   YAKE!
*   **Backend:**
    *   FastAPI
    *   Uvicorn
    *   Docker
*   **Frontend:**
    *   Streamlit
*   **Other:**
    *   NLTK (for `punkt` tokenizer, used by some summarizers)
    *   Requests

## Setup and Usage

There are three main components: the ML experimentation notebook, the FastAPI backend, and the Streamlit frontend.

### 1. Local ML Experimentation (Jupyter Notebook)

This is for developing and testing the core ML logic.

*   **Prerequisites:**
    *   Python 3.10 (or your compatible ARM64 version for M2 Mac)
    *   A virtual environment manager (e.g., `venv`)
*   **Setup:**
    1.  Navigate to the `content_analyzer_project` directory.
    2.  Create and activate a virtual environment:
        ```bash
        python3.10 -m venv venv  # Or your specific python command
        source venv/bin/activate
        ```
    3.  Install dependencies (ensure `tensorflow-macos` and `tensorflow-metal` versions are appropriate for your M2 Mac if listed here, or install them separately first as per detailed setup for M2):
        ```bash
        pip install --upgrade pip
        # For M2 Mac, install these first if not handled perfectly by requirements.txt:
        # pip install tensorflow-macos==<YOUR_WORKING_VERSION>
        # pip install tensorflow-metal==<YOUR_WORKING_VERSION>
        pip install -r requirements.txt
        pip install ipykernel
        python -m ipykernel install --user --name=content_analyzer_tf --display-name "Python (Content Analyzer TF)"
        ```
    4.  Download NLTK data (if not done previously):
        ```python
        # In a Python interpreter within the venv
        import nltk
        nltk.download('punkt')
        exit()
        ```
    5.  Start JupyterLab:
        ```bash
        jupyter lab
        ```
    6.  Open `notebooks/text_summarization_and_tagging.ipynb` and select the correct kernel. Run the cells to see the ML logic in action.

### 2. FastAPI Backend API

This API serves the summarization and tagging models.

**Option A: Run Locally with Uvicorn (using the same venv as the notebook)**

1.  Ensure the virtual environment from step 1 is active.
2.  Install API-specific dependencies (if not already covered by main `requirements.txt`):
    ```bash
    pip install fastapi uvicorn[standard] python-multipart
    ```
3.  Navigate to the `content_analyzer_project/api/` directory.
4.  Run the Uvicorn server:
    ```bash
    uvicorn main:app --reload --port 8002
    ```
    The API will be available at `http://localhost:8002` and docs at `http://localhost:8002/docs`. Models will be loaded on startup.

**Option B: Run with Docker**

1.  Ensure Docker Desktop is running.
2.  Modify `content_analyzer_project/api/requirements_api.txt` to use standard `tensorflow` (e.g., `tensorflow==<BASE_TF_VERSION>`) instead of `tensorflow-macos`.
3.  Navigate to the `content_analyzer_project/api/` directory.
4.  Build the Docker image:
    ```bash
    docker build -t content-analyzer-api .
    ```
5.  Run the Docker container:
    ```bash
    docker run -d -p 8002:8002 --name my-content-analyzer content-analyzer-api
    ```
    The API will be available at `http://localhost:8002`. The first time the container runs, it will download the Hugging Face models. Check logs with `docker logs my-content-analyzer`.

### 3. Streamlit Frontend UI

This UI interacts with the FastAPI backend.

1.  Ensure your FastAPI backend (from step 2A or 2B) is running and accessible on `http://localhost:8002`.
2.  Ensure the virtual environment from step 1 is active.
3.  Install Streamlit (if not already covered):
    ```bash
    pip install streamlit requests
    ```
4.  Navigate to the `content_analyzer_project` root directory (where `streamlit_app.py` is located).
5.  Run the Streamlit app:
    ```bash
    streamlit run streamlit_app.py
    ```
    The UI will open in your browser, typically at `http://localhost:8501`.

## Potential Future Enhancements

*   More robust error handling in API and UI.
*   Improved tagging using zero-shot classification models.
*   User authentication for the API.
*   Caching API responses in the Streamlit UI.
*   Deployment of API and UI to cloud platforms.
*   Option to upload documents instead of pasting text.

