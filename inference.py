# inference.py

import torch
import torchvision.transforms as T
from PIL import Image
import os
from generator import Generator

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_GEN = "saved_models/generator.pth"
INPUT_IMAGE_PATH = "input/lr_image.JPG"
OUTPUT_IMAGE_PATH = "output/hr_output.JPG"

# --- INFERENCE FUNCTION ---
def upscale_image(image_path, output_path):
    # 1. Load the trained generator model
    model = Generator().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_GEN, map_location=DEVICE))
    model.eval()

    # 2. Load and transform the low-resolution input image
    image = Image.open(image_path).convert("RGB")
    
    # The transform should ONLY normalize the image to [-1, 1]
    # The ToTensor() part handles scaling to [0, 1] first.
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # 3. Generate the high-resolution image
    with torch.no_grad():
        sr_tensor = model(input_tensor)

    # 4. De-normalize and save the output
    # The model outputs in [-1, 1], so convert it back to [0, 1] for saving
    sr_tensor = sr_tensor.squeeze(0).cpu()
    sr_tensor = sr_tensor.clamp(-1, 1) * 0.5 + 0.5 # De-normalize
    
    sr_image = T.ToPILImage()(sr_tensor)

    # 5. Save the final image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sr_image.save(output_path)
    print(f"✅ Super-resolution image saved to: {output_path}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    upscale_image(INPUT_IMAGE_PATH, OUTPUT_IMAGE_PATH)