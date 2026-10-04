import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(ROOT, "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from core.models.sign_cnn import SignCNN
from data.synthetic_signs import SyntheticSignDataset


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    full = SyntheticSignDataset(length=15000)
    val_size = 1500
    train_ds, val_ds = random_split(full, [len(full) - val_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)

    model = SignCNN(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 5
    for epoch in range(epochs):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
        print(f"[Epoch {epoch+1}] train loss={loss_sum/total:.4f} acc={correct/total:.4f}")

        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(1)
                correct += (preds == y).sum().item()
                total += x.size(0)
        print(f"[Epoch {epoch+1}] val acc={correct/total:.4f}")

    out_path = os.path.join(ROOT, "checkpoints", "sign_cnn.pt")
    torch.save(model.state_dict(), out_path)
    print(f"Modelo guardado en {out_path}")


if __name__ == "__main__":
    main()