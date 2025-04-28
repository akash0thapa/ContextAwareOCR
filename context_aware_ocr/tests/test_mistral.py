from llm_classifier.mistral_infer import MistralClassifier

ocr_output = """
Invoice Number: 9876
Date: 2024-03-20
Customer: Sita Rai
Total Amount: $200.00
Address: Kathmandu, Nepal
"""

classifier = MistralClassifier()
result = classifier.classify_text(ocr_output)
print("\n🧠 Mistral Output:\n", result)
