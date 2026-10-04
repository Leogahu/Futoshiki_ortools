import os
import sys
import json
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)

from core.vision.grid_detection import detect_grid, infer_n
from core.vision.cell_extraction import (
    extract_cells, extract_horizontal_edges, extract_vertical_edges,
)
from core.classification.predict import Predictor
from core.state.build_state import build_state


def montage(patches, cell_size):
    """
    Monta una lista de patches en una sola imagen para visualizarlos.
    patches: array (N, H, W) o (R, C, H, W)
    """
    if patches.ndim == 4:
        R, C, H, W = patches.shape
        canvas = np.zeros((R * H, C * W), dtype=np.uint8)
        for i in range(R):
            for j in range(C):
                canvas[i*H:(i+1)*H, j*W:(j+1)*W] = patches[i, j]
        return canvas
    elif patches.ndim == 3:
        N, H, W = patches.shape
        canvas = np.zeros((H, N * W), dtype=np.uint8)
        for i in range(N):
            canvas[:, i*W:(i+1)*W] = patches[i]
        return canvas
    return patches


def main(image_path):
    project_root = os.path.dirname(ROOT)
    digit_ckpt = os.path.join(project_root, "checkpoints", "digit_cnn.pt")
    sign_ckpt = os.path.join(project_root, "checkpoints", "sign_cnn.pt")

    if not os.path.exists(digit_ckpt) or not os.path.exists(sign_ckpt):
        print("Faltan los checkpoints.")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"No se pudo cargar: {image_path}")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"Imagen original: {gray.shape}")

    board_gray, rows_lines, cols_lines, _ = detect_grid(gray, debug=True)
    n = infer_n(rows_lines, cols_lines)
    print(f"Recorte del tablero: {board_gray.shape}")
    print(f"n inferido: {n}")

    if n < 2:
        print("No se detecto tablero valido.")
        return

    cells = extract_cells(board_gray, rows_lines, cols_lines, n)
    h_edges = extract_horizontal_edges(board_gray, rows_lines, cols_lines, n)
    v_edges = extract_vertical_edges(board_gray, rows_lines, cols_lines, n)
    print(f"Celdas: {cells.shape}")
    print(f"Bordes horizontales: {h_edges.shape}")
    print(f"Bordes verticales:   {v_edges.shape}")

    # --- GUARDAR PARCHES PARA DEBUG ---
    out_dir = os.path.join(project_root, "output")
    os.makedirs(out_dir, exist_ok=True)

    # montaje de celdas
    cell_size = cells.shape[2]
    grid_img = np.zeros((n * cell_size, n * cell_size), dtype=np.uint8)
    for i in range(n):
        for j in range(n):
            grid_img[i*cell_size:(i+1)*cell_size,
                     j*cell_size:(j+1)*cell_size] = cells[i, j]
    cv2.imwrite(os.path.join(out_dir, "patches_cells.png"),
                cv2.resize(grid_img, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))

    # montaje de bordes horizontales
    m_h = montage(h_edges, cell_size)
    cv2.imwrite(os.path.join(out_dir, "patches_hedges.png"),
                cv2.resize(m_h, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))

    # montaje de bordes verticales
    m_v = montage(v_edges, cell_size)
    cv2.imwrite(os.path.join(out_dir, "patches_vedges.png"),
                cv2.resize(m_v, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))

    print(f"Parches guardados en {out_dir}")
    # --- FIN DEBUG ---

    predictor = Predictor(digit_ckpt, sign_ckpt)
    grid = predictor.classify_digits(cells)
    h_pred = predictor.classify_horizontal_edges(h_edges)
    v_pred = predictor.classify_vertical_edges(v_edges)

    print("\n--- Grid detectado ---")
    for row in grid:
        print("  " + " ".join(str(int(x)) for x in row))

    print("\n--- Bordes horizontales ---")
    for row in h_pred:
        print("  " + " ".join(str(int(x)) for x in row))

    print("\n--- Bordes verticales ---")
    for row in v_pred:
        print("  " + " ".join(str(int(x)) for x in row))

    state = build_state(n, grid, h_pred, v_pred)

    print("\n--- JSON del estado inicial ---")
    print(json.dumps(state, indent=2, ensure_ascii=False))

    out_path = os.path.join(out_dir, "estado_inicial.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    print(f"\nJSON guardado en {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python src\\test_classify.py <ruta_imagen>")
        sys.exit(1)
    main(sys.argv[1])