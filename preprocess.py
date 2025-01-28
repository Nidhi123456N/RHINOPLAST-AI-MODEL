
 import numpy as np

from PIL import Image

# Preprocess the uploaded image
def preprocess_image(uploaded_file):
    """
    Convert the uploaded image to a PIL.Image.Image and resize to 512x512.
    """
    try:
        # Open the image using PIL
        img = Image.open(uploaded_file).convert("RGB")
        # Resize to 512x512 (required by the model)
        img_resized = img.resize((512, 512))
        return img_resized
    except Exception as e:
        raise ValueError(f"Error processing the image: {e}")




