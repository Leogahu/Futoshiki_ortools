import os
import sys
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)

from core.vision.grid_detection import detect_grid, infer_n
from core.vision.cell_extraction import (
    extract_cells, extract_horizontal_edges, extract_vertical_edges,
)

def main(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"No se pudo cargar: {image_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"Imagen original: {gray.shape}")

    # 1) detectar tablero y sus lineas
    board_gray, rows_lines, cols_lines, binary = detect_grid(gray, debug=True)
    print(f"Recorte del tablero: {board_gray.shape}")
    print(f"Lineas horizontales detectadas: {len(rows_lines)}")
    print(f"  posiciones Y: {[round(y, 1) for y in rows_lines]}")
    print(f"Lineas verticales detectadas:   {len(cols_lines)}")
    print(f"  posiciones X: {[round(x, 1) for x in cols_lines]}")

    n = infer_n(rows_lines, cols_lines)
    print(f"n inferido: {n}")

    if n < 2:
        print("No se detecto un tablero valido. Revisar debug_board.png")
        cv2.imwrite("debug_board.png", board_gray)
        cv2.imwrite("debug_binary.png", binary)
        return

    # 2) visualizar lineas sobre el tablero recortado
    canvas = cv2.cvtColor(board_gray, cv2.COLOR_GRAY2BGR)
    for y in rows_lines:
        cv2.line(canvas, (0, int(y)), (canvas.shape[1], int(y)), (0, 0, 255), 1)
    for x in cols_lines:
        cv2.line(canvas, (int(x), 0), (int(x), canvas.shape[0]), (255, 0, 0), 1)

    # 3) extraer celdas y bordes SOBRE EL RECORTE
    cells = extract_cells(board_gray, rows_lines, cols_lines, n)
    h_edges = extract_horizontal_edges(board_gray, rows_lines, cols_lines, n)
    v_edges = extract_vertical_edges(board_gray, rows_lines, cols_lines, n)

    print(f"Celdas: shape={cells.shape}")
    print(f"Bordes horizontales: shape={h_edges.shape}")
    print(f"Bordes verticales:   shape={v_edges.shape}")

    # 4) montajes visuales
    cell_size = cells.shape[2]
    grid_img = np.zeros((n * cell_size, n * cell_size), dtype=np.uint8)
    for i in range(n):
        for j in range(n):
            grid_img[i * cell_size:(i + 1) * cell_size,
                     j * cell_size:(j + 1) * cell_size] = cells[i, j]

    eh, ew = h_edges.shape[2], h_edges.shape[3]
    h_montage = np.zeros((n * eh, (n - 1) * ew), dtype=np.uint8)
    for i in range(n):
        for j in range(n - 1):
            h_montage[i * eh:(i + 1) * eh, j * ew:(j + 1) * ew] = h_edges[i, j]

    vh, vw = v_edges.shape[2], v_edges.shape[3]
    v_montage = np.zeros(((n - 1) * vh, n * vw), dtype=np.uint8)
    for i in range(n - 1):
        for j in range(n):
            v_montage[i * vh:(i + 1) * vh, j * vw:(j + 1) * vw] = v_edges[i, j]

    # 5) guardar debug
    out_dir = os.path.join(ROOT, "..", "output")
    os.makedirs(out_dir, exist_ok=True)

    def upscale(im, factor):
        hh, ww = im.shape[:2]
        return cv2.resize(im, (ww * factor, hh * factor),
                          interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(os.path.join(out_dir, "debug_board.png"), board_gray)
    cv2.imwrite(os.path.join(out_dir, "debug_lines.png"), canvas)
    cv2.imwrite(os.path.join(out_dir, "debug_cells.png"), upscale(grid_img, 3))
    cv2.imwrite(os.path.join(out_dir, "debug_hedges.png"), upscale(h_montage, 3))
    cv2.imwrite(os.path.join(out_dir, "debug_vedges.png"), upscale(v_montage, 3))
    print(f"Debug guardado en {out_dir}")

    # 6) mostrar
    cv2.imshow("tablero recortado", board_gray)
    cv2.imshow("lineas sobre el recorte", canvas)
    cv2.imshow("celdas (x3)", upscale(grid_img, 3))
    print("Presiona cualquier tecla para cerrar...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()