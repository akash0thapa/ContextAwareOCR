from preprocessing.image_cleaner import ImagePreprocessor

img_path = "data/sample_invoice.jpg"  # <- Replace with any image path
preprocessor = ImagePreprocessor()
cleaned_img_path = preprocessor.preprocess(img_path)

print("✅ Cleaned image saved at:", cleaned_img_path)
