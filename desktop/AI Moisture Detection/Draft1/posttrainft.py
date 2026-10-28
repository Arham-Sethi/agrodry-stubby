import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt

# ======== CONFIGURATION ========
DATASET_PATH = "path/to/your/parali_images"     
MODEL_PATH   = "moistnet_state_dict.pth"       
BATCH_SIZE   = 16

# ======== DEVICE SETUP ========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======== TRANSFORMS (MATCH TRAINING) ========
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet stats used during training
        std= [0.229, 0.224, 0.225]
    )
])

# ======== CUSTOM DATASET ========
class ParaliDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = [
            os.path.join(dp, f)
            for dp, _, files in os.walk(root_dir)
            for f in files
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img  = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, path

# ======== MODEL LOADING ========
def load_model(state_dict_path):
    # 1. Rebuild the architecture
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 1)  # single-value regression
    # 2. Load weights
    state = torch.load(state_dict_path, map_location=DEVICE)
    model.load_state_dict(state)
    # 3. Move to device & eval mode
    model.to(DEVICE).eval()
    return model

# ======== INFERENCE ========
def predict(model, dataloader):
    results = []
    with torch.no_grad(): # Disable gradient calculations for inference
        for imgs, paths in dataloader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs).squeeze(1)          
            preds   = outputs.cpu().tolist()          
            for p, m in zip(paths, preds):
                results.append((p, round(m, 2)))
    return results

# ======== MAIN ========
if __name__ == "__main__":
    # 1. Dataset & Loader
    ds = ParaliDataset(DATASET_PATH, transform=transform)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Load model
    model = load_model(MODEL_PATH)

    # 3. Run prediction
    preds = predict(model, loader)

    # 4. Print
    print("\n📋 Moisture Predictions:")
    for path, m in preds:
        print(f"{os.path.basename(path)} ➔ {m}%")

    # 5. Visualize first 5
    for path, m in preds[:5]:
        img = Image.open(path)
        plt.imshow(img)
        plt.title(f"{os.path.basename(path)} — {m}% moisture")
        plt.axis("off")
        plt.show()
        
    else: 
        print("/nFUCK! The model has been successfully FISTED!")
        
    