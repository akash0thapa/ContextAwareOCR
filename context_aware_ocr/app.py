import streamlit as st
from PIL import Image
import tempfile
import os
import json

# Import your unified pipeline
from pipeline.unified_inference import ContextAwareOCRPipeline

# ---------------------- STREAMLIT UI ----------------------

st.set_page_config(page_title="Unified OCR-LLM", layout="centered")
st.title("📄 Unified OCR + LLM Document Classifier")
st.markdown("Upload an image to extract structured data using OCR + Mistral-7B.")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.info("📦 Running unified pipeline...")

    # Initialize and run the pipeline
    try:
        pipeline = ContextAwareOCRPipeline()
        results = pipeline.process_image(tmp_path)

        # Show preprocessed image
        st.subheader("🧼 Preprocessed Image")
        st.image(results["cleaned_image"], caption="Cleaned Input", use_column_width=True)

        # Show OCR extracted text
        st.subheader("📝 Extracted Text")
        ocr_text = results["ocr_text"]
        st.text_area("OCR Output", ocr_text, height=250)

        # Show Mistral-structured output
        st.subheader("🧠 Mistral-7B Structured Classification")
        structured_output = results["structured_output"]

        try:
            parsed_json = json.loads(structured_output)
            st.json(parsed_json)
        except json.JSONDecodeError:
            st.text(structured_output)

    except Exception as e:
        st.error(f"❌ Pipeline failed: {e}")
    finally:
        os.remove(tmp_path)






