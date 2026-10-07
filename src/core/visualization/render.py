"""
Render visual de la solucion del Futoshiki sobre la imagen original.

Dibuja los numeros resueltos en las celdas vacias, con un color distinto
al de las pistas originales.
"""
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# fuentes comunes de Windows
FONT_PATHS = [
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/calibrib.ttf",
    "C:/Windows/Fonts/verdanab.ttf",
    "C:/Windows/Fonts/tahomabd.ttf",
    "C:/Windows/Fonts/segoeuib.ttf",
]


def _load_font(size: int):
    for path in FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def render_solution(imagen_bgr: np.ndarray,
                    state: dict,
                    solution: list,
                    color_solucion=(255, 100, 0),
                    debug: bool = False) -> np.ndarray:
    """
    Dibuja la solucion sobre la imagen original.

    Parametros:
        imagen_bgr      : imagen original en BGR (uint8)
        state           : estado inicial (JSON de la Fase 1)
        solution        : matriz n x n con la solucion
        color_solucion  : color BGR para los numeros resueltos
        debug           : si True, guarda imagen de debug

    Devuelve:
        imagen con la solucion superpuesta (np.ndarray BGR)
    """
    if solution is None:
        return imagen_bgr.copy()

    n = state["size"]
    grid = state["grid"]
    rows_lines = state["_meta"]["rows_lines"]
    cols_lines = state["_meta"]["cols_lines"]

    from core.vision.grid_detection import find_board_bbox, binarize

    gray = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    binary = binarize(gray)
    bbox = find_board_bbox(binary)
    if bbox is None:
        offset_x, offset_y = 0, 0
    else:
        x0, y0, x1, y1 = bbox
        offset_x = max(0, x0 - 2)
        offset_y = max(0, y0 - 2)

    # convertir a PIL para dibujar texto con fuentes TrueType
    img_pil = Image.fromarray(cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    for i in range(n):
        y0 = rows_lines[2 * i]
        y1 = rows_lines[2 * i + 1]
        for j in range(n):
            # si la celda ya tenia una pista original, no la sobreescribimos
            if grid[i][j] != 0:
                continue

            x0 = cols_lines[2 * j]
            x1 = cols_lines[2 * j + 1]

            # centro de la celda en coordenadas de la imagen original
            cx = int((x0 + x1) / 2) + offset_x
            cy = int((y0 + y1) / 2) + offset_y

            # tamano de fuente proporcional a la celda
            cell_size = int(min(x1 - x0, y1 - y0))
            font_size = int(cell_size * 0.55)
            font = _load_font(font_size)

            text = str(solution[i][j])

            # medir el texto para centrarlo
            bbox_text = draw.textbbox((0, 0), text, font=font)
            tw = bbox_text[2] - bbox_text[0]
            th = bbox_text[3] - bbox_text[1]

            tx = cx - tw // 2 - bbox_text[0]
            ty = cy - th // 2 - bbox_text[1]

            # dibujar texto en color de solucion (BGR -> RGB para PIL)
            color_rgb = (color_solucion[2], color_solucion[1], color_solucion[0])
            draw.text((tx, ty), text, fill=color_rgb, font=font)

    # volver a BGR
    result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    if debug:
        cv2.imwrite("debug_render.png", result)

    return result