# Step 2.1.1: Import required libraries
import cv2
import numpy as np
from PIL import Image
import os

# Step 2.1.2: Define a class for image cleaning
class ImagePreprocessor:
    def __init__(self, output_dir="output/cleaned_images"):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

    def preprocess(self, image_path):
        # Step 2.1.3: Load image in grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Step 2.1.4: Resize to standard width if too large
        max_width = 1000
        if img.shape[1] > max_width:
            scale_ratio = max_width / img.shape[1]
            img = cv2.resize(img, None, fx=scale_ratio, fy=scale_ratio, interpolation=cv2.INTER_AREA)

        # Step 2.1.5: Apply Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(img, (5, 5), 0)

        # Step 2.1.6: Binarize image using adaptive threshold
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # Step 2.1.7: Skew correction (deskewing)
        binary = self.deskew(binary)

        # Step 2.1.8: Save cleaned image
        filename = os.path.basename(image_path)
        cleaned_path = os.path.join(self.output_dir, f"cleaned_{filename}")
        cv2.imwrite(cleaned_path, binary)

        return cleaned_path

    def deskew(self, image):
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        (h, w) = image.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        deskewed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return deskewed
