import os
import numpy as np
import cv2
import torch

from core.vision.grid_detection import detect_grid, infer_n
from core.vision.cell_extraction import (
    extract_cells, extract_horizontal_edges, extract_vertical_edges,
)
from core.classification.predict import Predictor
from core.state.build_state import build_state


def _get_project_root():
    """
    Devuelve la ruta raiz del proyecto (Futoshilki/).
    """
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(here))


_PREDICTOR_CACHE = None


def get_predictor(device: str = None) -> Predictor:
    """
    Devuelve un Predictor ya inicializado (cacheado).
    """
    global _PREDICTOR_CACHE
    if _PREDICTOR_CACHE is None:
        root = _get_project_root()
        digit_ckpt = os.path.join(root, "checkpoints", "digit_cnn.pt")
        sign_ckpt = os.path.join(root, "checkpoints", "sign_cnn.pt")

        if not os.path.exists(digit_ckpt) or not os.path.exists(sign_ckpt):
            raise FileNotFoundError(
                f"No se encontraron los checkpoints en {root}/checkpoints/"
            )

        _PREDICTOR_CACHE = Predictor(digit_ckpt, sign_ckpt, device=device)

    return _PREDICTOR_CACHE


def procesar_imagen(imagen_gray: np.ndarray, debug: bool = False) -> dict:
    """
    Pipeline completo de la Fase 1: de imagen a estado inicial.

    Parametros:
        imagen_gray : np.ndarray en escala de grises (H, W) uint8.
        debug       : si True, imprime info adicional.

    Devuelve:
        dict con 'size', 'grid', 'horizontal_constraints', 'vertical_constraints'.
    """
    if imagen_gray is None or imagen_gray.size == 0:
        raise ValueError("Imagen vacia")

    # 1) deteccion de cuadricula
    board_gray, rows_lines, cols_lines = detect_grid(imagen_gray, debug=False)
    n = infer_n(rows_lines, cols_lines)

    if n < 2:
        raise RuntimeError(
            f"No se detecto un tablero valido (n inferido = {n})"
        )

    if debug:
        print(f"[pipeline] tablero {n}x{n} detectado")
        print(f"[pipeline] posiciones Y: {[round(y,1) for y in rows_lines]}")
        print(f"[pipeline] posiciones X: {[round(x,1) for x in cols_lines]}")

    # 2) extraccion de celdas y bordes
    cells = extract_cells(board_gray, rows_lines, cols_lines, n)
    h_edges = extract_horizontal_edges(board_gray, rows_lines, cols_lines, n)
    v_edges = extract_vertical_edges(board_gray, rows_lines, cols_lines, n)

    # 3) clasificacion
    predictor = get_predictor()
    grid = predictor.classify_digits(cells)
    h_pred = predictor.classify_horizontal_edges(h_edges)
    v_pred = predictor.classify_vertical_edges(v_edges)

    # 4) construccion del estado
    state = build_state(n, grid, h_pred, v_pred)
    return state