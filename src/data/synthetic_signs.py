import random
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset

SIZE = 64
CLASSES = ["nada", "<", ">"]


def _render_sign(label: int) -> np.ndarray:
    """
    Genera un parche cuadrado con:
    - fondo blanco
    - a veces lineas grises horizontales y/o verticales (simulando bordes de celda)
    - a veces un signo < > centrado o ligeramente descentrado
    """
    img = Image.new("L", (SIZE, SIZE), color=255)
    draw = ImageDraw.Draw(img)

    # 1) lineas grises simulando bordes de celda
    if random.random() < 0.6:
        y = random.choice([0, 1, SIZE - 2, SIZE - 1])
        gray_val = random.randint(120, 200)
        thickness = random.choice([1, 2])
        draw.line([(0, y), (SIZE, y)], fill=gray_val, width=thickness)
    if random.random() < 0.6:
        x = random.choice([0, 1, SIZE - 2, SIZE - 1])
        gray_val = random.randint(120, 200)
        thickness = random.choice([1, 2])
        draw.line([(x, 0), (x, SIZE)], fill=gray_val, width=thickness)

    # 2) signo (si label != 0)
    if label != 0:
        cx = SIZE // 2 + random.randint(-6, 6)
        cy = SIZE // 2 + random.randint(-6, 6)
        size = random.randint(18, 26)
        thickness = random.choice([2, 3, 4])

        if label == 1:   # "<"
            pts = [(cx + size // 2, cy - size),
                   (cx - size // 2, cy),
                   (cx + size // 2, cy + size)]
        else:            # ">"
            pts = [(cx - size // 2, cy - size),
                   (cx + size // 2, cy),
                   (cx - size // 2, cy + size)]

        draw.line(pts, fill=0, width=thickness, joint="curve")

    arr = np.array(img, dtype=np.float32) / 255.0

    if random.random() < 0.5:
        noise = np.random.normal(0, 0.03, arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0, 1).astype(np.float32)

    return arr.astype(np.float32)


class SyntheticSignDataset(Dataset):
    """Dataset al vuelo: signos con lineas de tablero simuladas."""
    def __init__(self, length=20000):
        self.length = length
        self.labels = []
        for _ in range(length):
            r = random.random()
            if r < 0.4:
                self.labels.append(0)
            elif r < 0.7:
                self.labels.append(1)
            else:
                self.labels.append(2)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        label = self.labels[idx]
        arr = _render_sign(label)
        tensor = torch.from_numpy(arr).unsqueeze(0).float()
        return tensor, label