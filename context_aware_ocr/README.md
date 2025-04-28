# Unified Framework for Context-Aware OCR and Text Classification using Large Language Models

This project integrates OCR with a locally-running LLM to:
- Extract text from scanned documents
- Understand and classify the text semantically (e.g., Invoice Number, Date, Total)

## 🧠 Technologies
- Donut OCR (Transformers)
- Mistral-7B (via LMStudio)
- Streamlit (UI)

## 🚀 How to Run
1. Clone the repo
2. Install requirements: `pip install -r requirements.txt`
3. Make sure LMStudio is running with API enabled
4. Run the app: `streamlit run app.py`

## 📂 Project Structure
...
context_aware_ocr_llm/
│
├── app.py                              # Streamlit frontend app
├── requirements.txt                   # Python dependencies
│
├── sample_docs/                       # Demo images
│   └── sample_invoice.jpg
│
├── preprocessing/                     # Image preprocessing module
│   └── image_cleaner.py
│
├── ocr_engine/                        # OCR model integration
│   └── donut_module.py
│
├── llm_classifier/                    # Mistral-7B (LLM) classification
│   └── mistral_infer.py
│
├── pipeline/                          # Unified pipeline connecting all modules
│   └── unified_inference.py
│
├── tests/                             # Test scripts (optional)
│   ├── test_donut.py
│   ├── test_mistral.py
│   └── test_pipeline.py
│
└── README.md                          # Project overview and instructions
