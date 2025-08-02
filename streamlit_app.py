# content_analyzer_project/streamlit_app.py
import streamlit as st
import requests  # To make HTTP requests to our FastAPI backend
import json      # To handle JSON data if needed for debugging (optional)

# --- Configuration ---
# This should be the address where your FastAPI backend is running.
# If FastAPI is running locally (uvicorn main:app --port 8002):
API_BASE_URL = "http://localhost:8002"
# If FastAPI is running in Docker and port 8002 is mapped to host's 8002:
# API_BASE_URL = "http://localhost:8002" # Still localhost from Streamlit's perspective

# --- Page Configuration (Optional, but nice) ---
st.set_page_config(
    page_title="Smart Content Analyzer",
    page_icon="📝",  # You can use an emoji or a URL to an icon
    layout="wide",   # Use "centered" or "wide"
    initial_sidebar_state="expanded" # "auto", "expanded", "collapsed"
)

# --- Main Application UI ---
st.title("📝 Smart Content Analyzer")
st.markdown("""
Welcome! Paste your text below, and this tool will use AI to:
- Generate a concise **summary**.
- Extract relevant **tags/keywords**.
""")
st.markdown("---") # Adds a horizontal line

# --- Sidebar for Controls (Optional) ---
st.sidebar.header("⚙️ Controls")
# These values will be passed to the API if your API's payload supports them
min_summary_len = st.sidebar.slider(
    "Minimum Summary Length (words)",
    min_value=10,
    max_value=100,
    value=30,  # Default value
    step=5
)
max_summary_len = st.sidebar.slider(
    "Maximum Summary Length (words)",
    min_value=50,
    max_value=500,
    value=150, # Default value
    step=10
)

# Ensure max is not less than min
if max_summary_len < min_summary_len:
    st.sidebar.warning("Max summary length should be >= min length. Adjusting max length.")
    max_summary_len = min_summary_len



# --- Input Text Area ---
st.subheader("📜 Input Your Text")
default_text = """Artificial intelligence (AI) is rapidly transforming various industries, from healthcare to finance and entertainment. Machine learning, a subset of AI, enables systems to learn from data and make predictions or decisions without being explicitly programmed. Deep learning, a further specialization, utilizes neural networks with many layers to analyze complex patterns in large datasets. Natural Language Processing (NLP) allows computers to understand, interpret, and generate human language, powering applications like chatbots and machine translation. Computer vision, another AI field, focuses on enabling machines to interpret and understand visual information from the world, such as images and videos. The ethical implications of AI, including bias in algorithms and job displacement, are critical areas of ongoing discussion and research. As AI technology continues to advance, its integration into daily life is expected to grow, offering both opportunities and challenges."""
input_text = st.text_area(
    "Paste or type your content here:",
    value=default_text,
    height=300,
    key="input_text_area" # Unique key for the widget
)

# --- Action Button ---
st.markdown("---")
if st.button("✨ Analyze Content", type="primary", use_container_width=True):
    if not input_text.strip():
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        # Show a spinner while processing
        with st.spinner("🧠 AI is thinking... Please wait..."):
            # Prepare the payload for the API
            api_payload = {
                "text": input_text,
                "max_summary_length": max_summary_len,
                "min_summary_length": min_summary_len
            }

            try:
                # Make the POST request to your FastAPI's /analyze endpoint
                analyze_url = f"{API_BASE_URL}/analyze"
                
                
                response = requests.post(analyze_url, json=api_payload, timeout=180) # Increased timeout for potentially long summarization

                # Check if the request was successful
                response.raise_for_status()  # This will raise an HTTPError for bad responses (4XX or 5XX)

                # Get the JSON response
                results = response.json()

                # --- Display Results ---
                st.subheader("📊 Analysis Results")

                # Using columns for a nicer layout
                col_summary, col_tags = st.columns(2)

                with col_summary:
                    st.markdown("#### Summarized Text:")
                    st.info(results.get("summary", "No summary was generated or an error occurred."))

                with col_tags:
                    st.markdown("#### Suggested Tags/Keywords:")
                    tags = results.get("tags", [])
                    if tags and not ("Error:" in tags[0] if isinstance(tags[0], str) else False) : # Check if tags list is not empty and first item isn't an error message
                        # Using st.chip for nicer tag display (requires Streamlit 1.30+)
                        # If you have an older Streamlit, use st.markdown for each tag.
                        chips_html = "".join([f'<span style="display: inline-block; background-color: #e0e0e0; color: #333; padding: 5px 10px; margin: 3px; border-radius: 15px; font-size: 0.9em;">🏷️ {tag}</span>' for tag in tags])
                        st.markdown(chips_html, unsafe_allow_html=True)
                        # Alternative for older Streamlit or simpler display:
                        # for tag in tags:
                        # st.markdown(f"- {tag}")
                    elif tags and "Error:" in tags[0]:
                         st.error(f"Tagging failed: {tags[0]}")
                    else:
                        st.markdown("_No tags were extracted._")
                
                # Optionally display the original text length for comparison
                if "original_text" in results: # Assuming API returns this
                    st.markdown(f"Original text length (words): {len(results['original_text'].split())}")


            except requests.exceptions.Timeout:
                st.error("⚠️ API Timeout: The analysis took too long to respond. Please try with shorter text or check the API server.")
            except requests.exceptions.ConnectionError:
                st.error(f"⚠️ API Connection Error: Could not connect to the API at {API_BASE_URL}. Please ensure the API server is running correctly.")
            except requests.exceptions.HTTPError as http_err:
                st.error(f"⚠️ API HTTP Error: {http_err}")
                try:
                    error_detail = response.json().get("detail", "No specific error detail from API.")
                    st.error(f"API Error Detail: {error_detail}")
                except json.JSONDecodeError:
                    st.error("API did not return valid JSON error details.")
            except Exception as e:
                st.error(f"An unexpected error occurred while processing: {e}")
                # For debugging, you might want to print the full response content if it's not JSON
                # if 'response' in locals() and response is not None:
                #     st.text_area("Raw API Response (for debugging):", response.text, height=100)

# --- Footer (Optional) ---
st.markdown("---")
st.markdown("Powered by Hugging Face Transformers, YAKE!, FastAPI, and Streamlit.")