# 🧩 Futoshiki Solver

Sistema **end-to-end** que lee una imagen de un acertijo **Futoshiki**, extrae su estado inicial mediante **Visión Computacional y Deep Learning**, y lo resuelve de manera óptima usando **Programación con Restricciones (CP)**.

El proyecto integra tres áreas:

1. **Visión Computacional + IA** — detección automática del tablero, clasificación de dígitos y signos con redes neuronales convolucionales.
2. **Constraint Programming** — modelado formal del puzzle y resolución con OR-Tools CP-SAT.
3. **Aplicación web** — interfaz amigable con Streamlit que permite subir una imagen o usar la cámara, y visualiza la solución sobre la imagen original.

---

## 📖 ¿Cómo se usa?

### Requisitos previos

- **Python 3.10 o superior** (probado con 3.11.9).
- **pip** actualizado.
- Opcional: **GPU NVIDIA** con CUDA 12.1 para acelerar el entrenamiento (no es obligatorio para resolver puzzles).

### Instalación

1. **Clona el repositorio:**

   ```bash
   git clone https://github.com/Leogahu/Futoshiki_ortools.git
   cd Futoshiki_ortools
   ```

2. **Crea un entorno virtual (recomendado):**

   ```bash
   python -m venv venv
   # En Windows:
   .\venv\Scripts\Activate.ps1
   # En Linux/Mac:
   source venv/bin/activate
   ```

3. **Instala las dependencias:**

   ```bash
   python -m pip install -r requirements.txt
   ```

   Si tienes GPU NVIDIA con CUDA 12.1 y quieres acelerar el entrenamiento:

   ```bash
   python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

### Ejecutar la aplicación web

La forma más simple de usar el sistema:

```bash
streamlit run apps/gui.py
```

Se abrirá el navegador en `http://localhost:8501`. Desde ahí puedes:

- **Subir una imagen** del Futoshiki (JPG o PNG).
- **O tomar una foto** con la cámara del dispositivo.
- Presionar **"Resolver puzzle"** y ver la solución superpuesta sobre la imagen original.
- Descargar el resultado como **JSON** o como **imagen PNG**.

### Usar desde la línea de comandos

Si prefieres la CLI:

```bash
# Extraer el estado inicial (JSON)
python apps/cli.py data/test1.png

# Especificar la ruta de salida
python apps/cli.py data/test1.png -o output/mi_resultado.json

# Con información de debug
python apps/cli.py data/test1.png --debug
```

Para resolver un JSON ya extraído:

```bash
python src/test_solver.py output/estado_inicial.json
```

Para el pipeline completo (imagen → solución):

```bash
python src/test_full.py data/test1.png
```

### Entrenar los modelos desde cero (opcional)

Los pesos ya están incluidos en `checkpoints/`. Si quieres reentrenar:

```bash
# CNN de dígitos
python src/train/train_digit.py

# CNN de signos
python src/train/train_sign.py
```

### Estructura del JSON de estado inicial

```json
{
  "size": 4,
  "grid": [
    [0, 0, 0, 1],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
  ],
  "horizontal_constraints": [
    [">", "<", null],
    [null, null, null],
    [null, null, null],
    [null, null, ">"]
  ],
  "vertical_constraints": [
    [null, null, null, null],
    [null, null, ">", ">"],
    [null, ">", null, null]
  ]
}
```

- `size`: tamaño del tablero (4 a 9).
- `grid`: matriz `n×n`, `0` = celda vacía, `1..n` = pista original.
- `horizontal_constraints`: matriz `n×(n-1)` con `"<"`, `">"` o `null`.
- `vertical_constraints`: matriz `(n-1)×n` con `"<"`, `">"` o `null`.

---

## 🛠️ ¿Qué se usó?

### Lenguajes y frameworks

| Tecnología | Uso |
|---|---|
| **Python 3.11** | Lenguaje principal |
| **PyTorch 2.5** | Entrenamiento de los CNNs (con CUDA 12.1) |
| **OpenCV 4.10** | Procesamiento de imagen y detección de cuadrícula |
| **OR-Tools CP-SAT** | Resolución del puzzle con constraint programming |
| **Streamlit** | Interfaz web interactiva |
| **NumPy** | Operaciones numéricas |
| **Pillow** | Manipulación de imágenes y render de texto |

### Arquitectura de los modelos

**DigitCNN** — clasifica el contenido de cada celda:

- 3 bloques `Conv2d + BatchNorm + ReLU + MaxPool`.
- `Flatten + Linear(128) + Dropout + Linear(10)`.
- Entrada: imagen en escala de grises de `64×64`.
- Salida: 10 clases (0 = vacío, 1–9 = dígito).

**SignCNN** — clasifica los signos de desigualdad:

- Misma estructura que `DigitCNN` pero con `AdaptiveAvgPool2d((6, 8))`.
- Entrada: imagen en escala de grises de `64×64`.
- Salida: 3 clases (0 = nada, 1 = `<`, 2 = `>`).
- Los signos verticales (`∧`, `∨`) se rotan 90° antes de clasificar.

Ambos modelos se entrenan con **datasets sintéticos generados al vuelo** que incluyen variaciones de fuente, tamaño, posición, ruido y brillo.

### Pipeline de Visión Computacional

1. **Preprocesamiento:** conversión a escala de grises y binarización con umbral de Otsu.
2. **Detección del tablero:** bounding box por proyección de perfil.
3. **Detección de líneas:** morfología horizontal y vertical + proyección de perfil con detección de picos. Cada celda está delimitada por **dos pares de líneas** (superior/inferior, izquierdo/derecho), por lo que se detectan `2n` líneas por eje.
4. **Auto-detección de `n`:** `n = (número de líneas por eje) / 2`.
5. **Extracción:** recorte de cada celda y de cada región entre celdas adyacentes (donde están los signos).
6. **Clasificación:** ambos CNNs clasifican cada recorte.
7. **Construcción del JSON:** empaquetado del estado inicial.

### Modelo de Constraint Programming

Variables y restricciones:

- **Variables:** `X[i][j]` con dominio `1..n`.
- **Restricciones globales:** `AllDifferent` por cada fila y por cada columna.
- **Restricciones binarias:** `X[i][j] < X[i][j+1]` o `>` según los signos horizontales.
- **Pistas:** `X[i][j] == valor` para celdas pre-rellenadas.
- **Solver:** OR-Tools CP-SAT, con `max_time_in_seconds` configurable.

### Tecnologías descartadas o consideradas

- **Tesseract OCR:** descartado por menor precisión con los signos `<` y `>`.
- **YOLO:** considerado, pero innecesario dado que la estructura del tablero es regular y OpenCV es suficiente.
- **Streamlit** vs. Gradio vs. Tkinter: se eligió Streamlit por su rapidez de desarrollo y soporte nativo de cámara.

---

## 📁 Estructura del repositorio

```
Futoshiki_ortools/
├── apps/                        # Capa de presentación
│   ├── cli.py                   # Interfaz de línea de comandos
│   └── gui.py                   # Interfaz web con Streamlit
│
├── checkpoints/                 # Pesos entrenados
│   ├── digit_cnn.pt
│   └── sign_cnn.pt
│
├── data/                        # Imágenes de prueba
│
├── output/                      # JSONs y resultados generados
│
├── src/
│   ├── core/                    # Lógica pura del sistema
│   │   ├── vision/              # Detección y extracción de imagen
│   │   ├── models/              # Arquitecturas CNN
│   │   ├── classification/      # Inferencia con los modelos
│   │   ├── state/               # Construcción del JSON
│   │   ├── cp/                  # Modelo y solver CP-SAT
│   │   ├── visualization/       # Render de la solución
│   │   └── pipeline.py          # Orquestador: imagen → solución
│   ├── data/                    # Generadores de datasets sintéticos
│   └── train/                   # Scripts de entrenamiento
│
├── .streamlit/config.toml       # Tema y configuración de Streamlit
├── requirements.txt
└── README.md
```

---

## 🎯 Estado del proyecto

| Fase | Descripción | Estado |
|------|-------------|--------|
| **1** | Visión Computacional + CNNs + JSON | ✅ Completada |
| **2** | Constraint Programming con OR-Tools | ✅ Completada |
| **3** | Integración, render visual y GUI | ✅ Completada |

---

## 📄 Licencia

Proyecto académico desarrollado para el curso **Tópicos de Ciencias de la Computación**. Uso libre para fines educativos.

---

## 👤 Autor

- **Leonardo G.** — [@Leogahu](https://github.com/Leogahu)