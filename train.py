# train.py 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
import os
from lpips import LPIPS

from generator import Generator
from discriminator import Discriminator


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_EPOCHS = 30
HR_DIR = "data/high_res/"
LR_DIR = "data/low_res/"
SAVE_MODEL_DIR = "saved_models/"
RESULTS_DIR = "results/"
CHECKPOINT_GEN = f"{SAVE_MODEL_DIR}generator.pth"
CHECKPOINT_DISC = f"{SAVE_MODEL_DIR}discriminator.pth"
HR_IMAGE_SIZE = (256, 256)
LR_IMAGE_SIZE = (64, 64)  # HR // 4


high_res_transform = T.Compose([
    T.Resize(HR_IMAGE_SIZE, interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])
low_res_transform = T.Compose([
    T.Resize(LR_IMAGE_SIZE, interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# --- DATASET ---
class ImageDataset(Dataset):
    def __init__(self, lr_dir, hr_dir):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.lr_images = sorted(os.listdir(lr_dir))
        self.hr_images = sorted(os.listdir(hr_dir))
        if len(self.lr_images) != len(self.hr_images):
            print(f"Warning: Image count mismatch (LR={len(self.lr_images)}, HR={len(self.hr_images)})")

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        lr_path = os.path.join(self.lr_dir, self.lr_images[idx])
        hr_path = os.path.join(self.hr_dir, self.hr_images[idx])
        lr = low_res_transform(Image.open(lr_path).convert("RGB"))
        hr = high_res_transform(Image.open(hr_path).convert("RGB"))
        return lr, hr

# --- TRAIN FUNCTION ---
def train_fn(loader, gen, disc, opt_gen, opt_disc, l1_loss, bce_loss, lpips_loss_fn):
    loop = tqdm(loader, leave=True)
    for lr, hr in loop:
        lr, hr = lr.to(DEVICE), hr.to(DEVICE)

        # Train Discriminator
        fake = gen(lr)
        disc_real = disc(hr)
        disc_fake = disc(fake.detach())
        loss_real = bce_loss(disc_real, torch.ones_like(disc_real))
        loss_fake = bce_loss(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_real + loss_fake) / 2
        
        opt_disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # Train Generator
        disc_fake = disc(fake)
        adv_loss = 1e-3 * bce_loss(disc_fake, torch.ones_like(disc_fake))
        content_loss = l1_loss(fake, hr)
        perceptual_loss = lpips_loss_fn(fake, hr).mean()
        loss_gen = content_loss + adv_loss + 0.1 * perceptual_loss

        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        loop.set_postfix(loss_disc=loss_disc.item(), loss_gen=loss_gen.item())

    # De-normalize images from [-1, 1] to [0, 1] for saving
    save_image(fake * 0.5 + 0.5, f"{RESULTS_DIR}/fake_epoch_end.png")
    save_image(hr * 0.5 + 0.5, f"{RESULTS_DIR}/real_epoch_end.png")

# --- MAIN ---
if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(SAVE_MODEL_DIR, exist_ok=True)

    dataset = ImageDataset(LR_DIR, HR_DIR)
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=8,  # Adjusted for common systems, change if needed
        pin_memory=True
    )

    gen = Generator().to(DEVICE)
    disc = Discriminator().to(DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))

    l1_loss = nn.L1Loss()
    bce_loss = nn.BCEWithLogitsLoss()

    lpips_loss_fn = LPIPS(net='alex').to(DEVICE) # Using 'alex' is standard and often faster
    for param in lpips_loss_fn.parameters():
        param.requires_grad = False

    print("\n✅ Training SRGAN with LPIPS for High-Quality Upscaling")
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch [{epoch+1}/{NUM_EPOCHS}] ---")
        train_fn(loader, gen, disc, opt_gen, opt_disc, l1_loss, bce_loss, lpips_loss_fn)
        torch.save(gen.state_dict(), CHECKPOINT_GEN)
        torch.save(disc.state_dict(), CHECKPOINT_DISC)

        print("✅ Checkpoints saved")
