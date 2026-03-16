# CV API — Clasificación de Imágenes con ResNet18

API REST de **visión por computadora** construida con **FastAPI** y **PyTorch**. Toma una imagen como entrada y devuelve la categoría que el modelo reconoce en ella, junto con el nivel de confianza de la predicción.

---

## ¿Qué hace este proyecto?

El sistema entrena un modelo de red neuronal convolucional (**ResNet18**) para reconocer objetos en imágenes y lo expone como una API web. El flujo completo es:

```
Usuario sube imagen  →  API procesa con ResNet18  →  Devuelve clase + confianza
```

Por ejemplo:
- Subes una foto de un **avión** → responde `"airplane"` con 94% de confianza
- Subes una foto de un **perro** → responde `"dog"` con 88% de confianza

### Clases que puede reconocer

El modelo fue entrenado con el dataset **CIFAR-10**, que contiene 10 categorías:

| Clase | Descripción |
|-------|-------------|
| `airplane` | Avión |
| `automobile` | Carro / Automóvil |
| `bird` | Pájaro |
| `cat` | Gato |
| `deer` | Venado |
| `dog` | Perro |
| `frog` | Rana |
| `horse` | Caballo |
| `ship` | Barco |
| `truck` | Camión |

> **Nota:** El modelo solo puede clasificar dentro de estas 10 categorías. Si subes una imagen de un objeto que no está en esta lista (un vaso, una silla, etc.), el modelo devolverá la categoría más parecida de las disponibles, pero no garantiza precisión. Ver la sección [Sugerencias para ampliar el modelo](#sugerencias-para-ampliar-el-modelo) al final de este documento.

---

## Arquitectura del proyecto

```
cv-api/
├── main.py          # Servidor FastAPI con los endpoints REST
├── inference.py     # Lógica de carga del modelo y predicción
├── train.py         # Script de entrenamiento del modelo
├── settings.py      # Configuración mediante variables de entorno
├── pyproject.toml   # Dependencias del proyecto
├── .env.example     # Plantilla de configuración
├── .env             # Configuración activa (no subir a Git)
├── data/            # Dataset CIFAR-10 (descargado automáticamente)
└── saved_models/
    └── resnet18_cifar10.pt  # Pesos del modelo entrenado
```

### Tecnologías utilizadas

| Tecnología | Rol |
|-----------|-----|
| **Python 3.12** | Lenguaje base |
| **PyTorch** | Framework de deep learning |
| **torchvision** | Dataset CIFAR-10 y modelo ResNet18 |
| **FastAPI** | Framework web para la API REST |
| **Uvicorn** | Servidor ASGI de alta performance |
| **Pydantic Settings** | Gestión de configuración mediante `.env` |
| **Pillow** | Procesamiento de imágenes |

---

## ¿Cómo funciona el modelo?

### ResNet18 con Transfer Learning

Se utiliza **ResNet18**, una red neuronal profunda con 18 capas que fue pre-entrenada en **ImageNet** (1.2 millones de imágenes, 1000 clases). En lugar de entrenar desde cero, se aplica **transfer learning** (fine-tuning):

1. Se toma ResNet18 con sus pesos pre-entrenados en ImageNet.
2. Se reemplaza la capa final (que tenía 1000 salidas) por una nueva con **10 salidas** (una por clase de CIFAR-10).
3. Se entrena el modelo completo con las imágenes de CIFAR-10 durante algunas épocas.

Este enfoque permite obtener un modelo preciso con mucho menos tiempo y datos que entrenar desde cero.

### Pipeline de preprocesamiento

Cada imagen que llega a la API pasa por el siguiente pipeline antes de ingresar al modelo:

```
Imagen original
    → Redimensionar a 256×256 px
    → Recorte central a 224×224 px   (tamaño que espera ResNet18)
    → Convertir a tensor
    → Normalizar con media y desviación estándar de ImageNet
    → Inferencia con el modelo
    → Softmax → probabilidades por clase
    → Retornar la clase con mayor probabilidad
```

---

## Instalación y configuración

### Requisitos previos

- Python 3.12+
- `pip` actualizado

### 1. Instalar dependencias

```powershell
pip install -e .
```

Esto instala todos los paquetes definidos en `pyproject.toml`:
- `fastapi[standard]`
- `uvicorn`
- `pydantic-settings`
- `torch` + `torchvision`
- `pillow`

### 2. Configurar variables de entorno

Copia la plantilla de configuración:

```powershell
copy .env.example .env
```

El archivo `.env` contiene:

```env
MODEL_PATH=saved_models/resnet18_cifar10.pt
UVICORN_HOST=127.0.0.1
UVICORN_PORT=8000
```

| Variable | Descripción | Valor por defecto |
|----------|-------------|-------------------|
| `MODEL_PATH` | Ruta al archivo de pesos del modelo | `saved_models/resnet18_cifar10.pt` |
| `UVICORN_HOST` | Host donde escucha el servidor | `0.0.0.0` |
| `UVICORN_PORT` | Puerto del servidor | `8000` |

---

## Entrenamiento del modelo

```powershell
python train.py
```

Este script realiza los siguientes pasos automáticamente:

1. **Descarga CIFAR-10** (~170 MB) en la carpeta `data/`
2. **Carga ResNet18** con pesos pre-entrenados de ImageNet
3. **Entrena el modelo** durante 2 épocas con optimizador Adam
4. **Evalúa la precisión** en el conjunto de prueba al final de cada época
5. **Guarda el modelo** en `saved_models/resnet18_cifar10.pt`

### Salida esperada durante el entrenamiento

```
Using device: cpu
Epoch 1/2 - loss: 0.8234 - test_acc: 0.7318
Epoch 2/2 - loss: 0.5912 - test_acc: 0.7856
Model saved to saved_models/resnet18_cifar10.pt
```

### Tiempo estimado de entrenamiento (CPU)

| Procesador | Tiempo por época | Total (2 épocas) |
|-----------|-----------------|-----------------|
| Intel i7 (11ª gen) | ~30-45 min | ~1-1.5 horas |
| Intel i5 (10ª gen) | ~45-60 min | ~1.5-2 horas |
| Con GPU (NVIDIA) | ~2-5 min | ~5-10 min |

> El modelo se entrena en **CPU** por defecto. Si tienes una GPU NVIDIA con CUDA instalado, PyTorch la detectará automáticamente y el entrenamiento será significativamente más rápido.

---

## Ejecutar la API

Con el modelo ya entrenado:

```powershell
python main.py
```

El servidor se levanta en:

- **API base:** `http://127.0.0.1:8000`
- **Documentación interactiva (Swagger UI):** `http://127.0.0.1:8000/docs`
- **Esquema OpenAPI (JSON):** `http://127.0.0.1:8000/openapi.json`
- **Página visual para subir imágenes:** `http://127.0.0.1:8000/upload`

---

## Endpoints de la API

### `GET /`
Mensaje de bienvenida. Confirma que el servidor está activo.

**Respuesta:**
```json
{
  "message": "Welcome to the CV API (ResNet18 on CIFAR-10)."
}
```

---

### `GET /upload`
Devuelve una página HTML muy sencilla con un formulario para subir una imagen desde el navegador (similar a un formulario de Flask).

1. Abre `http://127.0.0.1:8000/upload`
2. Elige un archivo de imagen
3. Haz clic en **Classify**

El formulario envía un `POST` a `/predict` con la imagen adjunta.

---

### `GET /health`
Verifica que el servidor está activo **y** que el modelo se puede cargar correctamente desde disco.

**Respuesta exitosa:**
```json
{
  "status": "ok",
  "model_path": "saved_models/resnet18_cifar10.pt"
}
```

**Respuesta si el modelo no existe todavía:**
```json
{
  "detail": "Error during prediction: ..."
}
```

---

### `POST /predict`
Recibe una imagen y devuelve la clase predicha por el modelo.

**Parámetros:**
- `file` *(multipart/form-data)*: Archivo de imagen (JPEG, PNG, etc.)

**Respuesta exitosa:**
```json
{
  "filename": "perro.jpg",
  "class_index": 5,
  "class_name": "dog",
  "probability": 0.8812
}
```

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `filename` | string | Nombre del archivo subido |
| `class_index` | int | Índice de la clase (0-9) |
| `class_name` | string | Nombre de la clase en inglés |
| `probability` | float | Confianza del modelo (0.0 - 1.0) |

**Errores posibles:**

| Código | Causa |
|--------|-------|
| `400` | El archivo no es una imagen o está vacío |
| `500` | Error interno durante la inferencia |

---

## Probar la API desde Swagger UI

1. Asegúrate de que el servidor esté corriendo con `python main.py`
2. Abre `http://127.0.0.1:8000/docs` en tu navegador
3. Haz clic en **`POST /predict`** → **Try it out**
4. En el campo `file`, selecciona una imagen desde tu computadora
5. Haz clic en **Execute**
6. Observa la respuesta JSON con la clase detectada y la probabilidad

---

## Probar la API desde consola (curl)

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@ruta/a/tu/imagen.jpg"
```

**Ejemplo de respuesta:**
```json
{
  "filename": "imagen.jpg",
  "class_index": 1,
  "class_name": "automobile",
  "probability": 0.9231
}
```

---

## Pasos completos (resumen rápido)

```powershell
# 1. Instalar dependencias
pip install -e .

# 2. Crear el archivo de configuración
copy .env.example .env

# 3. Entrenar el modelo (esperar ~1 hora en CPU)
python train.py

# 4. Levantar la API
python main.py

# 5. Probar desde el navegador
# Página visual para subir imágenes:
#   http://127.0.0.1:8000/upload
#
# (Opcional) Probar desde la documentación interactiva:
#   http://127.0.0.1:8000/docs
```

---

## Sugerencias para ampliar el modelo

El modelo actual reconoce únicamente las **10 clases de CIFAR-10**. Si necesitas detectar objetos más variados (vasos, muebles, personas, etc.), existen estas alternativas según el nivel de esfuerzo:

### Opción 1 — Usar ResNet18 con ImageNet (1,000 clases, sin entrenamiento)

ResNet18 ya viene pre-entrenado en ImageNet, que contiene **1,000 categorías** diferentes incluyendo objetos cotidianos. Para usar este modelo directamente (sin fine-tuning), bastaría con modificar `inference.py` para no reemplazar la capa final y utilizar las etiquetas de ImageNet.

**Ventaja:** No requiere entrenamiento, funciona de inmediato.  
**Desventaja:** Las 1,000 clases tienen nombres técnicos en inglés (p.ej. `"water_jug"`, `"coffee_mug"`) y no siempre son intuitivos.

### Opción 2 — Entrenar con un dataset propio

Si necesitas clases específicas (tu propio conjunto de objetos), puedes reemplazar CIFAR-10 por tus propias imágenes organizadas en carpetas por categoría y usar `ImageFolder` de torchvision. Necesitarías:
- Al menos **500-1000 imágenes** por clase
- Tiempo de entrenamiento proporcional al número de clases y tamaño del dataset

### Opción 3 — Modelos zero-shot (CLIP de OpenAI)

**CLIP** es un modelo de OpenAI que puede clasificar cualquier imagen en cualquier categoría definida en texto, sin necesidad de entrenamiento previo. Se le indica: *"¿Es esto un vaso, un carro o un edificio?"* y responde basándose en su comprensión visual y lingüística.

```bash
pip install transformers
```

Esta opción es la más flexible pero requiere más memoria RAM y tiempo de inferencia.

---

## Licencia

Este proyecto es de uso académico.
