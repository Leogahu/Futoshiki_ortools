"""
Solver para Futoshiki usando OR-Tools CP-SAT.
"""
from ortools.sat.python import cp_model

from core.cp.futoshiki_model import build_model


def resolver(state: dict, max_time_seconds: float = 30.0, verbose: bool = False):
    """
    Resuelve un Futoshiki a partir del estado inicial.

    Parametros:
        state           : dict con 'size', 'grid', 'horizontal_constraints',
                          'vertical_constraints'
        max_time_seconds: tiempo maximo de busqueda
        verbose         : si True, imprime estadisticas del solver

    Devuelve:
        dict con:
            'status'    : 'OPTIMAL', 'FEASIBLE', 'INFEASIBLE' o 'UNKNOWN'
            'solution'  : matriz n x n con la solucion (o None si no se encontro)
            'time'      : tiempo de resolucion en segundos
    """
    n = state["size"]
    model, X = build_model(state)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_time_seconds
    solver.parameters.num_search_workers = 1

    status = solver.Solve(model)
    status_name = solver.StatusName(status)

    result = {
        "status": status_name,
        "solution": None,
        "time": solver.WallTime(),
    }

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        solution = [[solver.Value(X[i][j]) for j in range(n)]
                    for i in range(n)]
        result["solution"] = solution

    if verbose:
        print(f"[solver] status: {status_name}")
        print(f"[solver] tiempo: {solver.WallTime():.3f} s")
        print(f"[solver] conflictos: {solver.NumConflicts()}")
        print(f"[solver] ramas: {solver.NumBranches()}")

    return result


def imprimir_solucion(state: dict, result: dict):
    """
    Imprime la solucion de forma legible: grid con signos entre celdas.
    """
    if result["solution"] is None:
        print(f"No se encontro solucion. Status: {result['status']}")
        return

    n = state["size"]
    sol = result["solution"]
    h = state["horizontal_constraints"]
    v = state["vertical_constraints"]

    for i in range(n):
        # linea de valores
        row_str = ""
        for j in range(n):
            val = sol[i][j]
            if j < n - 1:
                c = h[i][j] if h[i][j] else " "
                row_str += f" {val} {c}"
            else:
                row_str += f" {val}"
        print(row_str)

        # linea de signos verticales
        if i < n - 1:
            row_signs = ""
            for j in range(n):
                c = v[i][j] if v[i][j] else " "
                row_signs += f" {c}  "
            print(row_signs)