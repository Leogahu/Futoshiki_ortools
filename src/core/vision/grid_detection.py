import cv2
import numpy as np


def binarize(gray: np.ndarray) -> np.ndarray:
    """
    Binariza: lineas del tablero -> blanco (255), fondo -> negro.
    """
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh


def find_board_bbox(binary: np.ndarray):
    """
    Detecta el bounding box del tablero por proyeccion de perfil.
    """
    h, w = binary.shape
    row_sum = binary.sum(axis=1) / 255.0
    col_sum = binary.sum(axis=0) / 255.0

    thresh_row = 0.3 * w
    thresh_col = 0.3 * h

    rows_with_lines = np.where(row_sum > thresh_row)[0]
    cols_with_lines = np.where(col_sum > thresh_col)[0]

    if len(rows_with_lines) == 0 or len(cols_with_lines) == 0:
        return None

    y0 = int(rows_with_lines.min())
    y1 = int(rows_with_lines.max())
    x0 = int(cols_with_lines.min())
    x1 = int(cols_with_lines.max())

    if (x1 - x0) < 0.1 * w or (y1 - y0) < 0.1 * h:
        return None

    return x0, y0, x1, y1


def _smooth(profile: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Suaviza un vector 1D."""
    kernel = np.ones(ksize, dtype=np.float32) / ksize
    return np.convolve(profile, kernel, mode="same")


def _find_peaks(profile: np.ndarray,
                min_ratio: float = 0.3,
                min_distance: int = 15) -> list:
    """
    Encuentra los picos en un perfil 1D.
    """
    profile = _smooth(profile, ksize=5)
    max_val = profile.max()
    if max_val == 0:
        return []

    threshold = max_val * min_ratio
    above = profile > threshold

    positions = []
    i = 0
    n = len(profile)
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            segment = profile[i:j]
            weights = segment / segment.sum()
            center = (np.arange(i, j) * weights).sum()
            positions.append(float(center))
            i = j
        else:
            i += 1

    if len(positions) < 2:
        return positions

    # fusionar picos muy cercanos
    merged = [positions[0]]
    for p in positions[1:]:
        if p - merged[-1] < min_distance:
            merged[-1] = (merged[-1] + p) / 2
        else:
            merged.append(p)

    return merged


def detect_grid_on_board(board_gray: np.ndarray, debug: bool = False):
    """
    Detecta las lineas de la cuadricula SOBRE UN RECORTE del tablero.
    Cada celda esta delimitada por DOS pares de lineas.
    Devuelve 2*n coordenadas Y y 2*n coordenadas X.
    """
    binary = binarize(board_gray)

    if debug:
        cv2.imwrite("debug_binary_board.png", binary)

    k_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    horiz_only = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_horiz)

    k_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    vert_only = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_vert)

    if debug:
        cv2.imwrite("debug_horiz_only.png", horiz_only)
        cv2.imwrite("debug_vert_only.png", vert_only)

    row_profile = horiz_only.sum(axis=1) / 255.0
    col_profile = vert_only.sum(axis=0) / 255.0

    rows_peaks = _find_peaks(row_profile, min_ratio=0.3, min_distance=15)
    cols_peaks = _find_peaks(col_profile, min_ratio=0.3, min_distance=15)

    if debug:
        print(f"[grid] rows_peaks ({len(rows_peaks)}): "
              f"{[round(y,1) for y in rows_peaks]}")
        print(f"[grid] cols_peaks ({len(cols_peaks)}): "
              f"{[round(x,1) for x in cols_peaks]}")

    return rows_peaks, cols_peaks


def detect_grid(gray: np.ndarray, debug: bool = False):
    """
    Pipeline completo: bbox + deteccion de lineas.
    """
    binary = binarize(gray)
    bbox = find_board_bbox(binary)

    if bbox is None:
        if debug:
            print("[grid] no se encontro tablero, usando imagen completa")
        board_gray = gray
    else:
        x0, y0, x1, y1 = bbox
        if debug:
            print(f"[grid] bbox del tablero: ({x0}, {y0}) -> ({x1}, {y1})")
        pad = 2
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(gray.shape[1], x1 + pad)
        y1 = min(gray.shape[0], y1 + pad)
        board_gray = gray[y0:y1, x0:x1]

    if debug:
        rows_lines, cols_lines = detect_grid_on_board(board_gray, debug=True)
        return board_gray, rows_lines, cols_lines, None

    rows_lines, cols_lines = detect_grid_on_board(board_gray, debug=False)
    return board_gray, rows_lines, cols_lines


def infer_n(rows_lines: list, cols_lines: list) -> int:
    """
    Cada celda esta delimitada por DOS lineas. Entonces n = len // 2.
    """
    n_h = len(rows_lines) // 2
    n_v = len(cols_lines) // 2
    if n_h == n_v:
        return n_h
    return min(n_h, n_v)