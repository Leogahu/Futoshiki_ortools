import os
import numpy as np
import cv2
import torch

from core.models.digit_cnn import DigitCNN
from core.models.sign_cnn import SignCNN


DIGIT_CLASSES = ["vacio", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
SIGN_CLASSES = ["nada", "<", ">"]


class Predictor:
    """
    Carga los dos CNNs y expone metodos para clasificar celdas y bordes.
    """
    def __init__(self, digit_ckpt: str, sign_ckpt: str, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.digit_model = DigitCNN(num_classes=10).to(self.device)
        self.digit_model.load_state_dict(
            torch.load(digit_ckpt, map_location=self.device, weights_only=True)
        )
        self.digit_model.eval()

        self.sign_model = SignCNN(num_classes=3).to(self.device)
        self.sign_model.load_state_dict(
            torch.load(sign_ckpt, map_location=self.device, weights_only=True)
        )
        self.sign_model.eval()

    @staticmethod
    def _prepare_batch(patches: np.ndarray) -> torch.Tensor:
        arr = patches.astype(np.float32) / 255.0
        arr = arr[:, np.newaxis, :, :]
        return torch.from_numpy(arr)

    @torch.no_grad()
    def _classify_batch(self, model, batch_tensor) -> np.ndarray:
        batch_tensor = batch_tensor.to(self.device)
        logits = model(batch_tensor)
        return logits.argmax(dim=1).cpu().numpy()

    def classify_digits(self, cells: np.ndarray) -> np.ndarray:
        n = cells.shape[0]
        flat = cells.reshape(-1, cells.shape[2], cells.shape[3])
        tensor = self._prepare_batch(flat)
        preds = self._classify_batch(self.digit_model, tensor)
        return preds.reshape(n, n).astype(int)

    def classify_horizontal_edges(self, h_edges: np.ndarray) -> np.ndarray:
        n, m = h_edges.shape[0], h_edges.shape[1]
        flat = h_edges.reshape(-1, h_edges.shape[2], h_edges.shape[3])
        tensor = self._prepare_batch(flat)
        preds = self._classify_batch(self.sign_model, tensor)
        return preds.reshape(n, m).astype(int)

    def classify_vertical_edges(self, v_edges: np.ndarray) -> np.ndarray:
        n_rows, n_cols = v_edges.shape[0], v_edges.shape[1]
        rotated = np.zeros((n_rows, n_cols, v_edges.shape[3], v_edges.shape[2]),
                           dtype=np.uint8)
        for i in range(n_rows):
            for j in range(n_cols):
                rotated[i, j] = cv2.rotate(
                    v_edges[i, j], cv2.ROTATE_90_COUNTERCLOCKWISE
                )
        flat = rotated.reshape(-1, rotated.shape[2], rotated.shape[3])
        tensor = self._prepare_batch(flat)
        preds = self._classify_batch(self.sign_model, tensor)
        return preds.reshape(n_rows, n_cols).astype(int)