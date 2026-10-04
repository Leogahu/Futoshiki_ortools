import os
import sys
import json
import argparse
import cv2

# Agregar src/ al sys.path para poder importar "from core.pipeline import ..."
# cli.py esta en apps/, entonces subimos un nivel y entramos a src/
HERE = os.path.dirname(os.path.abspath(__file__))       # .../Futoshilki/apps
PROJECT_ROOT = os.path.dirname(HERE)                     # .../Futoshilki
SRC_DIR = os.path.join(PROJECT_ROOT, "src")              # .../Futoshilki/src

sys.path.append(SRC_DIR)

from core.pipeline import procesar_imagen


def main():
    parser = argparse.ArgumentParser(
        description="Futoshiki Solver - Fase 1: extrae el estado inicial de una imagen."
    )
    parser.add_argument("imagen", help="ruta a la imagen del Futoshiki (jpg/png)")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="ruta al JSON de salida (default: output/estado_inicial.json)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="muestra informacion adicional",
    )
    args = parser.parse_args()

    # cargar imagen
    img = cv2.imread(args.imagen)
    if img is None:
        print(f"Error: no se pudo cargar la imagen {args.imagen}")
        sys.exit(1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # procesar
    try:
        state = procesar_imagen(gray, debug=args.debug)
    except Exception as e:
        print(f"Error procesando la imagen: {e}")
        sys.exit(1)

    # mostrar por consola
    print(json.dumps(state, indent=2, ensure_ascii=False))

    # guardar
    if args.output is None:
        out_dir = os.path.join(PROJECT_ROOT, "output")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "estado_inicial.json")
    else:
        out_path = args.output
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

    print(f"\nJSON guardado en {out_path}")


if __name__ == "__main__":
    main()