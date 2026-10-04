import cv2
import numpy as np


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def extract_cells(gray: np.ndarray, rows_lines: list, cols_lines: list,
                  n: int, cell_size: int = 64, shrink: float = 0.15):
    """
    Recorta las n*n celdas del tablero.
    """
    cells = np.zeros((n, n, cell_size, cell_size), dtype=np.uint8)

    for i in range(n):
        y0_cell = rows_lines[2 * i]
        y1_cell = rows_lines[2 * i + 1]
        h_cell = y1_cell - y0_cell
        dy = int(h_cell * shrink)
        y0 = int(y0_cell) + dy
        y1 = int(y1_cell) - dy

        for j in range(n):
            x0_cell = cols_lines[2 * j]
            x1_cell = cols_lines[2 * j + 1]
            w_cell = x1_cell - x0_cell
            dx = int(w_cell * shrink)
            x0 = int(x0_cell) + dx
            x1 = int(x1_cell) - dx

            y0 = _clip(y0, 0, gray.shape[0] - 1)
            y1 = _clip(y1, y0 + 1, gray.shape[0])
            x0 = _clip(x0, 0, gray.shape[1] - 1)
            x1 = _clip(x1, x0 + 1, gray.shape[1])

            crop = gray[y0:y1, x0:x1]
            crop = cv2.resize(crop, (cell_size, cell_size),
                              interpolation=cv2.INTER_AREA)
            cells[i, j] = crop

    return cells


def _cell_size_avg(rows_lines: list, cols_lines: list, n: int) -> int:
    """
    Devuelve un tamano representativo de celda (promedio del alto y ancho).
    """
    heights = [rows_lines[2 * i + 1] - rows_lines[2 * i] for i in range(n)]
    widths = [cols_lines[2 * j + 1] - cols_lines[2 * j] for j in range(n)]
    avg_h = int(np.mean(heights))
    avg_w = int(np.mean(widths))
    return max(int((avg_h + avg_w) / 2), 20)


def _extract_centered_square(gray: np.ndarray, cx: int, cy: int,
                              side: int, out_size: int = 64) -> np.ndarray:
    """
    Recorta un cuadrado de lado `side` centrado en (cx, cy).
    Si se sale de la imagen, rellena con blanco.
    Redimensiona a out_size x out_size.
    """
    h, w = gray.shape
    half = side // 2

    x0 = cx - half
    y0 = cy - half
    x1 = x0 + side
    y1 = y0 + side

    # crear canvas blanco del tamano del cuadrado
    canvas = np.full((side, side), 255, dtype=np.uint8)

    # calcular interseccion con la imagen
    sx0 = max(0, x0)
    sy0 = max(0, y0)
    sx1 = min(w, x1)
    sy1 = min(h, y1)

    if sx0 >= sx1 or sy0 >= sy1:
        return cv2.resize(canvas, (out_size, out_size),
                          interpolation=cv2.INTER_AREA)

    # posicion dentro del canvas
    cx0 = sx0 - x0
    cy0 = sy0 - y0
    cx1 = cx0 + (sx1 - sx0)
    cy1 = cy0 + (sy1 - sy0)

    canvas[cy0:cy1, cx0:cx1] = gray[sy0:sy1, sx0:sx1]
    return cv2.resize(canvas, (out_size, out_size),
                      interpolation=cv2.INTER_AREA)


def extract_horizontal_edges(gray: np.ndarray, rows_lines: list,
                             cols_lines: list, n: int,
                             edge_size: int = 64):
    """
    Recorta los bordes horizontales (signos < >) entre celdas adyacentes.
    El signo esta en el punto medio entre la celda (i,j) y la celda (i,j+1),
    a la altura del centro vertical de la fila i.
    """
    edges = np.zeros((n, n - 1, edge_size, edge_size), dtype=np.uint8)
    cell_size = _cell_size_avg(rows_lines, cols_lines, n)
    half = cell_size // 2

    for i in range(n):
        # centro vertical de la celda (i, j)
        y_center = int((rows_lines[2 * i] + rows_lines[2 * i + 1]) / 2)

        for j in range(n - 1):
            # el signo esta en el punto medio entre celda j y celda j+1,
            # que es exactamente cols_lines[2*j+1] (borde derecho de celda j)
            # y cols_lines[2*j+2] (borde izquierdo de celda j+1).
            # Tomamos el punto medio entre esos dos bordes.
            x_center = int((cols_lines[2 * j + 1] + cols_lines[2 * j + 2]) / 2)

            patch = _extract_centered_square(gray, x_center, y_center,
                                             cell_size, edge_size)
            edges[i, j] = patch

    return edges


def extract_vertical_edges(gray: np.ndarray, rows_lines: list,
                           cols_lines: list, n: int,
                           edge_size: int = 64):
    """
    Recorta los bordes verticales (signos ^ v) entre celdas adyacentes.
    El signo esta en el punto medio entre la celda (i,j) y la celda (i+1,j),
    a la altura del centro horizontal de la columna j.
    """
    edges = np.zeros((n - 1, n, edge_size, edge_size), dtype=np.uint8)
    cell_size = _cell_size_avg(rows_lines, cols_lines, n)

    for i in range(n - 1):
        # el signo esta en el punto medio entre fila i y fila i+1,
        # que es el punto medio entre rows_lines[2*i+1] y rows_lines[2*i+2]
        y_center = int((rows_lines[2 * i + 1] + rows_lines[2 * i + 2]) / 2)

        for j in range(n):
            # centro horizontal de la celda (i, j) y (i+1, j)
            x_center = int((cols_lines[2 * j] + cols_lines[2 * j + 1]) / 2)

            patch = _extract_centered_square(gray, x_center, y_center,
                                             cell_size, edge_size)
            edges[i, j] = patch

    return edges