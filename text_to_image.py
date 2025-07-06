#pip install transformers torch accelerate
from diffusers import StableDiffusionPipeline
import torch

# Load the stable Diffusion Model
model_id = "CompVis/Stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype = torch.float16
)

#Move the model to GPU If Avaliable

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)
from diffusers import StableDiffusionPipeline
import torch

# Load the stable Diffusion Model
model_id = "CompVis/Stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    # torch_dtype = torch.float16 # Removed as per error message suggestion
)

#Move the model to GPU If Avaliable

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

# Define your text Prompt
prompt = "Young Indian scientist discovering a glowing energy crystal in a futuristic lab, sci-fi style, cinematic lighting."

# Generate the image
# with torch.autocast("cuda"):# Use autocast for mixed precision (Faster on GPU) # Commented out as float16 is removed
image = pipe(prompt).images[0]
# Save the Image
image.save("generated_image.png")
print("Image saved as generated_image.png")
