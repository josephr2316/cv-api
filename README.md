# CV API — Image Classification with ResNet18

REST API for **computer vision** built with **FastAPI** and **PyTorch**. It takes an image as input and returns the predicted class along with its confidence.

---

## What does this project do?

The system trains a convolutional neural network (**ResNet18**) to recognize objects in images and exposes it as a web API. The full flow is:

```
User uploads image  →  API processes it with ResNet18  →  Returns class + confidence
```

For example:
- You upload a photo of an **airplane** → it responds `"airplane"` with 94% confidence
- You upload a photo of a **dog** → it responds `"dog"` with 88% confidence

### Classes it can recognize

The model is trained on the **CIFAR-10** dataset, which contains 10 categories:

| Class | Description |
|-------|-------------|
| `airplane` | Airplane |
| `automobile` | Car / Automobile |
| `bird` | Bird |
| `cat` | Cat |
| `deer` | Deer |
| `dog` | Dog |
| `frog` | Frog |
| `horse` | Horse |
| `ship` | Ship / Boat |
| `truck` | Truck |

> **Note:** The model can only classify within these 10 categories. If you upload an image of an object that is not in this list (a cup, a chair, etc.), the model will still return the closest class among these 10, but the prediction may not be meaningful. See [Suggestions to extend the model](#suggestions-to-extend-the-model) at the end of this document.

---

## Project architecture

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

### Technologies used

| Technology | Role |
|-----------|-----|
| **Python 3.12** | Base language |
| **PyTorch** | Deep learning framework |
| **torchvision** | CIFAR-10 dataset and ResNet18 model |
| **FastAPI** | Web framework for the REST API |
| **Uvicorn** | High-performance ASGI server |
| **Pydantic Settings** | Configuration management via `.env` |
| **Pillow** | Image processing |

---

## How does the model work?

### ResNet18 with transfer learning

The project uses **ResNet18**, a deep convolutional network with 18 layers that was pre-trained on **ImageNet** (1.2M images, 1000 classes). Instead of training from scratch, we apply **transfer learning** (fine-tuning):

1. Start from ResNet18 with its pre-trained ImageNet weights.
2. Replace the final fully-connected layer (originally 1000 outputs) with a new one with **10 outputs** (one per CIFAR-10 class).
3. Train the whole model on CIFAR-10 images for a few epochs.

This approach gives good accuracy with far less data and time than training from scratch.

### Preprocessing pipeline

Every image that reaches the API goes through the following pipeline before entering the model:

```
Original image
    → Resize to 256×256 px
    → Center crop to 224×224 px   (size expected by ResNet18)
    → Convert to tensor
    → Normalize with ImageNet mean and std
    → Run inference with the model
    → Softmax → per-class probabilities
    → Return the class with the highest probability
```

---

## Installation and configuration

### Prerequisites

- Python 3.12+
- Recent `pip`

### 1. Install dependencies

```powershell
pip install -e .
```

This installs all packages defined in `pyproject.toml`:
- `fastapi[standard]`
- `uvicorn`
- `pydantic-settings`
- `torch` + `torchvision`
- `pillow`

### 2. Configure environment variables

Copy the example config:

```powershell
copy .env.example .env
```

The `.env` file contains:

```env
MODEL_PATH=saved_models/resnet18_cifar10.pt
UVICORN_HOST=127.0.0.1
UVICORN_PORT=8000
```

| Variable | Description | Default value |
|----------|-------------|---------------|
| `MODEL_PATH` | Path to the model weights file | `saved_models/resnet18_cifar10.pt` |
| `UVICORN_HOST` | Host where the server listens | `127.0.0.1` |
| `UVICORN_PORT` | Server port | `8000` |

---

## Model training

```powershell
python train.py
```

This script performs the following steps automatically:

1. **Downloads CIFAR-10** (~170 MB) into the `data/` folder
2. **Loads ResNet18** with ImageNet pre-trained weights
3. **Trains the model** for 2 epochs with the Adam optimizer
4. **Evaluates accuracy** on the test set after each epoch
5. **Saves the model** to `saved_models/resnet18_cifar10.pt`

### Expected training output

```
Using device: cpu
Epoch 1/2 - loss: 0.8234 - test_acc: 0.7318
Epoch 2/2 - loss: 0.5912 - test_acc: 0.7856
Model saved to saved_models/resnet18_cifar10.pt
```

### Estimated training time (CPU)

| Processor | Time per epoch | Total (2 epochs) |
|-----------|----------------|------------------|
| Intel i7 (11th gen) | ~30–45 min | ~1–1.5 hours |
| Intel i5 (10th gen) | ~45–60 min | ~1.5–2 hours |
| With GPU (NVIDIA) | ~2–5 min | ~5–10 min |

> The model is trained on **CPU** by default. If you have an NVIDIA GPU with CUDA installed, PyTorch will detect it automatically and training will be much faster.

---

## Run the API

With the model already trained:

```powershell
python main.py
```

The server runs at:

- **Base API:** `http://127.0.0.1:8000`
- **Interactive docs (Swagger UI):** `http://127.0.0.1:8000/docs`
- **OpenAPI schema (JSON):** `http://127.0.0.1:8000/openapi.json`
- **Visual upload page:** `http://127.0.0.1:8000/upload`

---

## API endpoints

### `GET /`
Welcome message. Confirms the server is running.

**Response:**
```json
{
  "message": "Welcome to the CV API (ResNet18 on CIFAR-10)."
}
```

---

### `GET /upload`
Returns an HTML page with a polished form to upload an image from the browser (similar to a Flask form, but styled and with drag & drop).

1. Open `http://127.0.0.1:8000/upload`
2. Choose an image file (or drag & drop it)
3. Click **Clasificar imagen / Classify**

The form sends a `POST` request to `/predict` with the image attached.

---

### `GET /health`
Checks that the server is running **and** that the model can be loaded correctly from disk.

**Successful response:**
```json
{
  "status": "ok",
  "model_path": "saved_models/resnet18_cifar10.pt"
}
```

**Response if the model does not exist yet:**
```json
{
  "detail": "Error during prediction: ..."
}
```

---

### `POST /predict`
Receives an image and returns the predicted class.

**Parameters:**
- `file` *(multipart/form-data)*: Image file (JPEG, PNG, etc.)

**Successful response:**
```json
{
  "filename": "perro.jpg",
  "class_index": 5,
  "class_name": "dog",
  "probability": 0.8812
}
```

| Field | Type | Description |
|-------|------|-------------|
| `filename` | string | Uploaded file name |
| `class_index` | int | Class index (0–9) |
| `class_name` | string | Class name (English) |
| `probability` | float | Model confidence (0.0–1.0) |

**Possible errors:**

| Code | Cause |
|------|-------|
| `400` | File is not an image or is empty |
| `500` | Internal error during inference |

---

## Test the API from Swagger UI

1. Make sure the server is running with `python main.py`
2. Open `http://127.0.0.1:8000/docs` in your browser
3. Click **`POST /predict`** → **Try it out**
4. In the `file` field, select an image from your computer
5. Click **Execute**
6. Inspect the JSON response for predicted class and probability

---

## Test the API from the console (curl)

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@ruta/a/tu/imagen.jpg"
```

**Example response:**
```json
{
  "filename": "imagen.jpg",
  "class_index": 1,
  "class_name": "automobile",
  "probability": 0.9231
}
```

---

## Full workflow (quick recap)

```powershell
# 1. Install dependencies
pip install -e .

# 2. Create the config file
copy .env.example .env

# 3. Train the model (wait ~1 hour on CPU)
python train.py

# 4. Run the API
python main.py

# 5. Test from the browser
# Visual page to upload images:
#   http://127.0.0.1:8000/upload
#
# (Optional) Test from the interactive documentation:
#   http://127.0.0.1:8000/docs
```

---

## Suggestions to extend the model

The current model only recognizes the **10 CIFAR-10 classes**. If you need to detect more diverse objects (cups, furniture, people, etc.), here are some options:

### Option 1 — Use ResNet18 with ImageNet (1,000 classes, no training)

ResNet18 comes pre-trained on ImageNet, which contains **1,000 different categories**, including many everyday objects. To use this model directly (without fine-tuning), you would modify `inference.py` so it does not replace the final layer and uses the original ImageNet labels.

**Pros:** No training required, works immediately.  
**Cons:** The 1,000 classes have technical English names (e.g. `"water_jug"`, `"coffee_mug"`) and are not always intuitive.

### Option 2 — Train with a custom dataset

If you need specific classes (your own object set), you can replace CIFAR-10 with your own images organized in folders per category and use `ImageFolder` from torchvision. You would typically need:
- At least **500–1000 images** per class
- Training time proportional to the number of classes and dataset size

### Option 3 — Zero-shot models (OpenAI CLIP)

**CLIP** is a model that can classify any image into any text-defined category, without additional training. You tell it: *“Is this a cup, a car, or a building?”* and it responds based on its joint visual–text understanding.

```bash
pip install transformers
```

This option is the most flexible but requires more RAM and longer inference times.

---

## License

This project is intended for academic use.
