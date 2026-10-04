import numpy as np


def build_state(n: int,
                grid: np.ndarray,
                h_pred: np.ndarray,
                v_pred: np.ndarray) -> dict:
    """
    Convierte las predicciones en el JSON del estado inicial.

    grid   : (n, n)       int, 0..9
    h_pred : (n, n-1)     int, 0=nada, 1="<", 2=">"
    v_pred : (n-1, n)     int, 0=nada, 1="<", 2=">"
    """
    grid_json = grid.astype(int).tolist()

    # bordes horizontales
    # h_pred[i][j] esta entre celda (i, j) y celda (i, j+1)
    h_constraints = []
    for i in range(n):
        row = []
        for j in range(n - 1):
            v = int(h_pred[i][j])
            if v == 1:
                row.append("<")
            elif v == 2:
                row.append(">")
            else:
                row.append(None)
        h_constraints.append(row)

    # bordes verticales
    # tras rotar antihorario:
    #   1 ("<") -> original era ^ (apunta arriba) -> arriba < abajo -> "<"
    #   2 (">") -> original era v (apunta abajo)  -> arriba > abajo -> ">"
    v_constraints = []
    for i in range(n - 1):
        row = []
        for j in range(n):
            v = int(v_pred[i][j])
            if v == 1:
                row.append("<")
            elif v == 2:
                row.append(">")
            else:
                row.append(None)
        v_constraints.append(row)

    return {
        "size": n,
        "grid": grid_json,
        "horizontal_constraints": h_constraints,
        "vertical_constraints": v_constraints,
    }