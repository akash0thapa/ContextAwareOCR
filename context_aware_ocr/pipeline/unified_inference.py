import pytesseract
from PIL import Image, ImageFilter, ImageOps
import requests
import json


class ContextAwareOCRPipeline:
    def __init__(self):
        self.mistral_url = "http://localhost:1234/v1/completions"
        self.model_name = "mathstral-7b-v0.1"

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert("L")  # grayscale
        image = ImageOps.invert(image)
        image = image.filter(ImageFilter.MedianFilter())
        return image

    def run_ocr(self, image):
        return pytesseract.image_to_string(image, lang='eng')

    def call_mistral(self, extracted_text):
        prompt = f"""
You are an intelligent document parser.
Analyze the given text and extract all useful information.
Return a structured JSON where keys are meaningful categories based on context, such as:
- Person details (e.g. name, title, phone, email)
- Organization info (company name, department, address)
- Document-specific data (license number, issue date, etc.)
- Any other relevant fields

Avoid fixed fields. Just extract and label everything sensibly.
Key Note: You should also use your logic in checking whether there is any spelling errors or any incorrect data interpretation..
Text to analyze:
\"\"\"
{extracted_text}
\"\"\"
Output JSON:
"""

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": 512,
            "temperature": 0.3,
        }

        try:
            response = requests.post(self.mistral_url, json=payload, timeout=180)
            result = response.json()

            if "choices" in result:
                response_text = result["choices"][0]["text"].strip()
                return response_text
            else:
                return f"❌ Mistral format unexpected: {result}"
        except Exception as e:
            return f"❌ Mistral call error: {e}"

    def process_image(self, image_path):
        # Step 1: Preprocess
        cleaned_image = self.preprocess_image(image_path)

        # Step 2: OCR
        extracted_text = self.run_ocr(cleaned_image)

        # Step 3: Mistral inference
        structured_output = self.call_mistral(extracted_text)

        return {
            "cleaned_image": cleaned_image,
            "ocr_text": extracted_text,
            "structured_output": structured_output
        }
