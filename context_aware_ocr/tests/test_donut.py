from ocr_engine.donut_module import DonutOCR

ocr = DonutOCR()
image_path = "output/cleaned_images/cleaned_sample_invoice.jpg"  # <- Use preprocessed image
text_output = ocr.extract_text(image_path)

print("\n📄 OCR Result:\n", text_output)
