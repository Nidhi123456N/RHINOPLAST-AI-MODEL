
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor

import torch

 # Convert to torch.Tensor
# Convert to numpy.ndarray

def load_pipeline():
    """
    Load the Stable Diffusion Inpainting pipeline with GPU support.
    """
    try:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16  # Mixed precision for GPU
        ).to("cuda")  # Use GPU
        return pipe
    except Exception as e:
        raise RuntimeError(f"Error loading pipeline: {e}")



def generate_image(pipe, prompt, img):
    """
    Generate the modified image based on the input image and prompt.
    """
    try:
        # Validate the input image type
        if not isinstance(img, Image.Image):
            raise ValueError(f"Input image must be of type PIL.Image.Image. Got: {type(img)}")

        # Convert the image to a numpy array (if required)
        img_array = np.array(img)

        # Convert the numpy array to a torch tensor (if required by pipeline)
        img_tensor = ToTensor()(img).unsqueeze(0)  # Add batch dimension

        # Pass the image to the pipeline (supports PIL, numpy, or torch.Tensor)
        result_image = pipe(prompt=prompt, image=img, mask_image=None).images[0]
        return result_image
    except Exception as e:
        raise RuntimeError(f"Error generating image: {e}")
