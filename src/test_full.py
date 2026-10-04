import os
import sys
import json
import time
import cv2

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)

from core.pipeline import resolver_imagen
from core.cp.solver import imprimir_solucion


def main(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"No se pudo cargar: {image_path}")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print(f"Imagen: {image_path}")
    print(f"Procesando...\n")

    from core.pipeline import resolver_imagen
    from core.visualization.render import render_solution

    t0 = time.time()
    result = resolver_imagen(gray, debug=True)
    total_time = time.time() - t0

    state = result["state"]
    solution = result["solution"]

    # render
    img_solution = render_solution(img, state, solution, debug=True)

    # mostrar
    print("\n--- Solucion ---")
    if solution is not None:
        imprimir_solucion(state, {"solution": solution, "status": result["solver_status"]})
    else:
        print(f"No se encontro solucion. Status: {result['solver_status']}")

    print(f"\n--- Resumen ---")
    print(f"Tamano: {result['n']}x{result['n']}")
    print(f"Status: {result['solver_status']}")
    print(f"Tiempo solver: {result['solver_time']:.4f} s")
    print(f"Tiempo total:  {total_time:.4f} s")

    # guardar
    project_root = os.path.dirname(ROOT)
    out_dir = os.path.join(project_root, "output")
    os.makedirs(out_dir, exist_ok=True)

    # guardar imagen con solucion
    out_img_path = os.path.join(out_dir, "solucion_visual.png")
    cv2.imwrite(out_img_path, img_solution)
    print(f"\nImagen con solucion guardada en {out_img_path}")

    # guardar JSON
    state_clean = {k: v for k, v in state.items() if k != "_meta"}
    output = {
        "state": state_clean,
        "solution": solution,
        "solver_status": result["solver_status"],
        "solver_time": result["solver_time"],
        "n": result["n"],
    }
    out_json_path = os.path.join(out_dir, "resultado_completo.json")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"JSON guardado en {out_json_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python src\\test_full.py <ruta_imagen>")
        sys.exit(1)
    main(sys.argv[1])