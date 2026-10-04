import os
import sys
import json

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)

from core.cp.solver import resolver, imprimir_solucion


def main(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    print(f"Tablero: {state['size']}x{state['size']}")
    print("\nEstado inicial:")
    imprimir_solucion(state, {"solution": state["grid"], "status": "INITIAL"})

    print("\nResolviendo...")
    result = resolver(state, verbose=True)

    print("\nSolucion:")
    imprimir_solucion(state, result)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python src\\test_solver.py <ruta_json>")
        sys.exit(1)
    main(sys.argv[1])