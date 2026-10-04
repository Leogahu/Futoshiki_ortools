"""
Modelo CP-SAT para Futoshiki.

Variables:
    X[i][j] en dominio 1..n

Restricciones:
    - AllDifferent por fila
    - AllDifferent por columna
    - Pistas: X[i][j] == valor
    - Desigualdades horizontales y verticales
"""
from ortools.sat.python import cp_model


def build_model(state: dict):
    """
    Construye el modelo CP-SAT a partir del estado inicial.

    Parametros:
        state : dict con 'size', 'grid', 'horizontal_constraints', 'vertical_constraints'

    Devuelve:
        (model, X) donde X es una matriz n x n de variables cp_model.IntVar.
    """
    n = state["size"]
    grid = state["grid"]
    h_constraints = state["horizontal_constraints"]
    v_constraints = state["vertical_constraints"]

    model = cp_model.CpModel()

    # 1) variables: X[i][j] en dominio 1..n
    X = [[model.NewIntVar(1, n, f"X_{i}_{j}") for j in range(n)]
         for i in range(n)]

    # 2) restricciones globales: alldifferent por fila y por columna
    for i in range(n):
        model.AddAllDifferent([X[i][j] for j in range(n)])
    for j in range(n):
        model.AddAllDifferent([X[i][j] for i in range(n)])

    # 3) pistas iniciales
    for i in range(n):
        for j in range(n):
            v = grid[i][j]
            if v != 0:
                model.Add(X[i][j] == v)

    # 4) restricciones de desigualdad horizontal
    # h_constraints[i][j] esta entre X[i][j] y X[i][j+1]
    for i in range(n):
        for j in range(n - 1):
            c = h_constraints[i][j]
            if c == "<":
                model.Add(X[i][j] < X[i][j + 1])
            elif c == ">":
                model.Add(X[i][j] > X[i][j + 1])

    # 5) restricciones de desigualdad vertical
    # v_constraints[i][j] esta entre X[i][j] y X[i+1][j]
    for i in range(n - 1):
        for j in range(n):
            c = v_constraints[i][j]
            if c == "<":
                model.Add(X[i][j] < X[i + 1][j])
            elif c == ">":
                model.Add(X[i][j] > X[i + 1][j])

    return model, X