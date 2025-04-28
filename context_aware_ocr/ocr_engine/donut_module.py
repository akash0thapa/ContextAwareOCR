# Step 2.2.1: Imports
import torch
from transformers import VisionEncoderDecoderModel, DonutProcessor
from PIL import Image

# Step 2.2.2: OCR Module with Donut
class DonutOCR:
    def __init__(self, model_name="naver-clova-ix/donut-base-finetuned-docvqa"):
        print("🔍 Loading Donut model... (may take a minute)")
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.processor = DonutProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def extract_text(self, image_path):
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)

        task_prompt = "<s_docvqa><s_question>What is written in this document?</s_question><s_answer>"
        decoder_input_ids = self.processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=512,
                early_stopping=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )

        result = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return result
