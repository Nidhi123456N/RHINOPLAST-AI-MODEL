from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import torch

# Load pipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16
).to("cuda")

# Load and preprocess test image
img = Image.open("test_image.jpg").convert("RGB").resize((512, 512))

# Generate image
prompt = "A photorealistic side profile of a person with a straighter nose"
result = pipe(prompt=prompt, image=img, mask_image=None).images[0]
result.save("test_output.png")
print("Test output saved as 'test_output.png'")



