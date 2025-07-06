#pip install transformers diffusers torch
from diffusers import StableDiffusionPipeline
import torch

# Load the model
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype = torch.float16)
pipe = pipe.to("cuda")
prompt = "A futuristic cityscape at night with flying boats. "
frames = []
for i in range(10):
  frame = pipe(prompt).images[0]
  frames.append(frame)
import cv2
import numpy as np

# Save frames as images
for i, frame in enumerate(frames):
    frame_np = np.array(frame)  # Ensure frame is a NumPy array
    cv2.imwrite(f"frame_{i}.png", frame_np)

# Set video parameters
frame_rate = 5
frame = cv2.imread("frame_0.png")  # Read the first frame to get size
frame_height, frame_width, _ = frame.shape
frame_size = (frame_width, frame_height)

# Create video writer
out = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, frame_size)

# Write all frames to video
for i in range(len(frames)):
    frame = cv2.imread(f"frame_{i}.png")
    out.write(frame)

out.release()
print("Video saved as output_video.mp4")
