import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from losses import ssim_vi, ssim_ir, RMI_ir, RMI_vi
from dataset import TrainData
from model import FAMAFuse


#       Reproducibility
def setup_seed(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}.")



# Train Step
def train_step(model, optimizer, weights, img1, img2, device):
    model.train()
    model.to(device)

    img1, img2 = img1.to(device), img2.to(device)
    img1 = (img1 - img1.min()) / (img1.max() - img1.min())
    img2 = (img2 - img2.min()) / (img2.max() - img2.min())

    fused = model(img1, img2)

    loss = (
            weights[0] * ssim_ir(fused, img2) +
            weights[1] * ssim_vi(fused, img1) +
            weights[2] * RMI_ir(fused, img1) +
            weights[3] * RMI_vi(fused, img2)
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return model, fused, loss.item()

class Config:
    # --- Configuration --- #
    EPOCHS = 50
    BATCH_SIZE = 8
    LR = 1e-3
    SEED = 3407
    IMG_SIZE = (256, 256)
    LOSS_WEIGHTS = [1, 1, 1, 1.25]
    FUSION_TYPE = "SPECT-MRI"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Paths --- #
    DATA_DIR = f"/kaggle/input/fusion/FUSION DATASET/{FUSION_TYPE}/train/"
    MODEL_DIR = f"./modelsave/{FUSION_TYPE}/{EPOCHS}"
    RESULT_DIR = f"./result/{FUSION_TYPE}"
    TEMP_DIR = f"./temp/{FUSION_TYPE}"
    MODEL_NAME = "mri_spect_weight.pth"

config = Config()
def main():


    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.RESULT_DIR, exist_ok=True)
    os.makedirs(config.TEMP_DIR, exist_ok=True)

    setup_seed(config.SEED)

    # --- Data Loader --- #
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(config.IMG_SIZE)
    ])
    train_set = TrainData(config.DATA_DIR, img_type1="SPECT/", img_type2="MRI/", transform=transform)
    train_loader = data.DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True,
                                   drop_last=True, num_workers=2, pin_memory=True)

    # --- Model, Optimizer, Scheduler --- #
    model = FAMAFuse(channels=1, out_channels=64).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                                           patience=5, verbose=True)

    best_loss = float('inf')
    loss_history = []

    # --- Training Loop --- #
    for epoch in range(config.EPOCHS):
        epoch_losses = []

        for img_ir, img_vr in train_loader:
            model, fused, loss = train_step(model, optimizer, config.LOSS_WEIGHTS, img_ir, img_vr, config.DEVICE)
            epoch_losses.append(loss)

        avg_loss = np.mean(epoch_losses)
        loss_history.append(avg_loss)
        scheduler.step(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, config.MODEL_NAME))
            print(f"[Epoch {best_epoch}] Best model saved with loss {best_loss:.4f}")

        print(f"[Epoch {epoch + 1}] Average Loss: {avg_loss:.5f}")

    # --- Plot Loss --- #
    plt.figure()
    plt.plot(range(1, config.EPOCHS + 1), loss_history, 'r-', label='Training Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss Curve - {config.FUSION_TYPE}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config.MODEL_DIR, "loss.png"))
    plt.show()


if __name__ == "__main__":
    main()
