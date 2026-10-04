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
from core.cp.solver import resolver


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
    Pipeline de la Fase 1: de imagen a estado inicial (JSON).
    """
    if imagen_gray is None or imagen_gray.size == 0:
        raise ValueError("Imagen vacia")

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

    cells = extract_cells(board_gray, rows_lines, cols_lines, n)
    h_edges = extract_horizontal_edges(board_gray, rows_lines, cols_lines, n)
    v_edges = extract_vertical_edges(board_gray, rows_lines, cols_lines, n)

    predictor = get_predictor()
    grid = predictor.classify_digits(cells)
    h_pred = predictor.classify_horizontal_edges(h_edges)
    v_pred = predictor.classify_vertical_edges(v_edges)

    state = build_state(n, grid, h_pred, v_pred)

    # guardamos metadatos utiles para la fase de visualizacion
    state["_meta"] = {
        "board_gray_shape": board_gray.shape,
        "rows_lines": rows_lines,
        "cols_lines": cols_lines,
    }
    return state


def resolver_imagen(imagen_gray: np.ndarray,
                     debug: bool = False,
                     max_time_seconds: float = 30.0) -> dict:
    """
    Pipeline completo: de imagen a solucion resuelta.

    Parametros:
        imagen_gray      : np.ndarray en escala de grises.
        debug            : si True, imprime info adicional.
        max_time_seconds : tiempo maximo de busqueda del solver.

    Devuelve dict con:
        {
            "state":         # estado inicial (JSON de Fase 1)
            "solution":      # matriz n x n con la solucion (o None)
            "solver_status": # 'OPTIMAL', 'FEASIBLE', 'INFEASIBLE', 'UNKNOWN'
            "solver_time":   # tiempo del solver en segundos
            "n":             # tamano del tablero
        }
    """
    # Fase 1
    state = procesar_imagen(imagen_gray, debug=debug)

    # Fase 2
    result = resolver(state, max_time_seconds=max_time_seconds, verbose=debug)

    return {
        "state": state,
        "solution": result["solution"],
        "solver_status": result["status"],
        "solver_time": result["time"],
        "n": state["size"],
    }

def resolver_y_renderizar(imagen_bgr: np.ndarray,
                           debug: bool = False) -> dict:
    """
    Pipeline completo con render visual.

    Devuelve dict con:
        - state
        - solution
        - solver_status
        - solver_time
        - n
        - imagen_solucion : imagen original con la solucion superpuesta
    """
    from core.visualization.render import render_solution

    gray = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    result = resolver_imagen(gray, debug=debug)

    imagen_solucion = render_solution(
        imagen_bgr,
        result["state"],
        result["solution"],
        debug=debug,
    )
    result["imagen_solucion"] = imagen_solucion
    return result