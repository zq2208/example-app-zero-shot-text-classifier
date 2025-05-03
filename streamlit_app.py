import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Set page configuration
st.set_page_config(
    layout="centered", page_title="OOV Sentiment Analysis"
)

# Initialize session state
if "valid_inputs_received" not in st.session_state:
    st.session_state["valid_inputs_received"] = False

############ CREATE THE LOGO AND HEADING ############
st.caption("")
st.title("OOV Sentiment Analysis")

############ SIDEBAR CONTENT ############
st.sidebar.write("")
MODEL_PATH = "./final_distilbert_model"
st.sidebar.markdown(f"Using model: DistilBERT from {MODEL_PATH}")
st.sidebar.markdown("---")

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Failed to load model or tokenizer: {str(e)}")
        return None, None

tokenizer, model = load_model_and_tokenizer()

if tokenizer is None or model is None:
    st.stop()

############ TABBED NAVIGATION ############
InfoTab, MainTab = st.tabs(["Info", "Main"])

with InfoTab:
    st.subheader("What is Streamlit?")
    st.markdown(
        "[Streamlit](https://streamlit.io) is a Python library that allows the creation of interactive, data-driven web applications in Python."
    )
    st.subheader("Resources")
    st.markdown(
        """
        - [Streamlit Documentation](https://docs.streamlit.io/)
        - [Cheat sheet](https://docs.streamlit.io/library/cheatsheet)
        - [Book](https://www.amazon.com/dp/180056550X) (Getting Started with Streamlit for Data Science)
        """
    )
    st.subheader("Deploy")
    st.markdown(
        "You can quickly deploy Streamlit apps using [Streamlit Community Cloud](https://streamlit.io/cloud) in just a few clicks."
    )

with MainTab:
    st.write("")
    st.write("")
    st.write("")

    with st.form(key="my_form"):
        # Text input for sentiment analysis
        input_text = st.text_area(
            "Enter text for sentiment analysis",
            placeholder="Type your text here...",
            height=150
        )
        submit_button = st.form_submit_button(label="Classify Sentiment")

    ############ CONDITIONAL STATEMENTS ############
    if not submit_button and not st.session_state.valid_inputs_received:
        st.stop()

    elif submit_button and not input_text.strip():
        st.warning("Please enter text to classify.")
        st.session_state.valid_inputs_received = False
        st.stop()

    elif submit_button or st.session_state.valid_inputs_received:
        if submit_button:
            st.session_state.valid_inputs_received = True

        ############ Modeling Call ############
        with st.spinner("Classifying..."):
            # Tokenize input
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                padding=True
            )
            # Inference
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                pred = torch.argmax(probs, dim=1).item()

            # Convert label {0, 1} to {-1, 1} as per your model's evaluation
            label_map = {0: -1, 1: 1}
            sentiment = label_map[pred]

            # Display results
            st.success(f"Predicted Sentiment: {sentiment} ({'Positive' if sentiment == 1 else 'Negative'})")
            st.write(f"Confidence: {probs[0][pred].item():.2f}")
