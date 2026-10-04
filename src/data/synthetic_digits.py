import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import Dataset

IMG_SIZE = 64
CLASSES = ["vacio", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

# Fuentes comunes de Windows. Si alguna no existe, se ignora.
FONT_PATHS = [
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/calibrib.ttf",
    "C:/Windows/Fonts/verdanab.ttf",
    "C:/Windows/Fonts/tahomabd.ttf",
    "C:/Windows/Fonts/segoeuib.ttf",
]


def _random_font(size):
    for _ in range(10):
        path = random.choice(FONT_PATHS)
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _render_cell(label: int) -> np.ndarray:
    """Genera una celda 64x64 con un digito o vacia."""
    img = Image.new("L", (IMG_SIZE, IMG_SIZE), color=255)
    draw = ImageDraw.Draw(img)

    # borde opcional (algunas veces)
    if random.random() < 0.7:
        draw.rectangle(
            [0, 0, IMG_SIZE - 1, IMG_SIZE - 1],
            outline=0,
            width=random.choice([1, 2]),
        )

    if label != 0:
        text = str(label)
        font = _random_font(random.randint(30, 44))
        bbox = draw.textbbox((0, 0), text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (IMG_SIZE - w) / 2 - bbox[0] + random.randint(-3, 3)
        y = (IMG_SIZE - h) / 2 - bbox[1] + random.randint(-3, 3)
        draw.text((x, y), text, fill=0, font=font)

    arr = np.array(img, dtype=np.float32) / 255.0

    # augmentations simples (forzando float32 para no promover a float64)
    if random.random() < 0.5:
        noise = np.random.normal(0, 0.03, arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 1).astype(np.float32)
    if random.random() < 0.3:
        arr = np.clip(arr * random.uniform(0.85, 1.15), 0, 1).astype(np.float32)

    return arr.astype(np.float32)


class SyntheticDigitDataset(Dataset):
    """Dataset al vuelo: no guarda nada en disco."""
    def __init__(self, length=20000):
        self.length = length
        self.labels = []
        for _ in range(length):
            if random.random() < 0.2:
                self.labels.append(0)          # vacio
            else:
                self.labels.append(random.randint(1, 9))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        label = self.labels[idx]
        arr = _render_cell(label)
        tensor = torch.from_numpy(arr).unsqueeze(0).float()   # (1, 64, 64) float32
        return tensor, label