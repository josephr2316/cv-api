from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from inference import load_model, predict_image_bytes
from settings import settings


app = FastAPI(
    title="CV API - ResNet18 CIFAR-10",
    description=(
        "Computer vision API that fine-tunes a ResNet18 model on CIFAR-10 and exposes "
        "a /predict endpoint to classify uploaded images."
    ),
)


@app.get("/")
async def root():
    return {"message": "Welcome to the CV API (ResNet18 on CIFAR-10)."}


UPLOAD_PAGE_HTML = """
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>CV API — Clasificación de imágenes</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&display=swap" rel="stylesheet" />
  <style>
    :root {
      --bg: #0f172a;
      --surface: #1e293b;
      --surface-hover: #334155;
      --border: #475569;
      --text: #f1f5f9;
      --text-muted: #94a3b8;
      --accent: #6366f1;
      --accent-hover: #818cf8;
      --success: #22c55e;
      --error: #ef4444;
      --radius: 12px;
      --shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'DM Sans', system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 1.5rem;
      line-height: 1.5;
    }
    .container {
      width: 100%;
      max-width: 480px;
    }
    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 2rem;
      box-shadow: var(--shadow);
    }
    h1 {
      font-size: 1.5rem;
      font-weight: 700;
      margin-bottom: 0.25rem;
      letter-spacing: -0.02em;
    }
    .subtitle {
      color: var(--text-muted);
      font-size: 0.9375rem;
      margin-bottom: 1.75rem;
    }
    .dropzone {
      border: 2px dashed var(--border);
      border-radius: var(--radius);
      padding: 2rem;
      text-align: center;
      cursor: pointer;
      transition: border-color 0.2s, background 0.2s;
      margin-bottom: 1rem;
      position: relative;
    }
    .dropzone:hover, .dropzone.dragover {
      border-color: var(--accent);
      background: rgba(99, 102, 241, 0.08);
    }
    .dropzone input[type="file"] {
      position: absolute;
      inset: 0;
      opacity: 0;
      cursor: pointer;
      width: 100%;
      height: 100%;
    }
    .dropzone-icon {
      font-size: 2.5rem;
      margin-bottom: 0.5rem;
      opacity: 0.9;
    }
    .dropzone-text {
      font-weight: 500;
      color: var(--text);
    }
    .dropzone-hint {
      font-size: 0.8125rem;
      color: var(--text-muted);
      margin-top: 0.25rem;
    }
    .file-name {
      font-size: 0.875rem;
      color: var(--accent);
      margin-top: 0.75rem;
      word-break: break-all;
    }
    .btn {
      width: 100%;
      padding: 0.875rem 1.25rem;
      font-family: inherit;
      font-size: 1rem;
      font-weight: 600;
      color: white;
      background: var(--accent);
      border: none;
      border-radius: var(--radius);
      cursor: pointer;
      transition: background 0.2s;
    }
    .btn:hover:not(:disabled) {
      background: var(--accent-hover);
    }
    .btn:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }
    .result {
      margin-top: 1.5rem;
      padding: 1.25rem;
      border-radius: var(--radius);
      background: rgba(34, 197, 94, 0.12);
      border: 1px solid rgba(34, 197, 94, 0.3);
      display: none;
    }
    .result.error {
      background: rgba(239, 68, 68, 0.12);
      border-color: rgba(239, 68, 68, 0.3);
    }
    .result.visible { display: block; }
    .result-title {
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
      margin-bottom: 0.5rem;
    }
    .result-class {
      font-size: 1.25rem;
      font-weight: 700;
      color: var(--success);
    }
    .result.error .result-class { color: var(--error); }
    .result-prob {
      font-size: 0.9375rem;
      color: var(--text-muted);
      margin-top: 0.25rem;
    }
    .spinner {
      display: inline-block;
      width: 1rem;
      height: 1rem;
      border: 2px solid rgba(255,255,255,0.3);
      border-top-color: white;
      border-radius: 50%;
      animation: spin 0.7s linear infinite;
      vertical-align: -0.2em;
      margin-right: 0.5rem;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .footer {
      margin-top: 1.5rem;
      text-align: center;
      font-size: 0.8125rem;
      color: var(--text-muted);
    }
    .footer a { color: var(--accent); text-decoration: none; }
    .footer a:hover { text-decoration: underline; }
  </style>
</head>
<body>
  <div class="container">
    <div class="card">
      <h1>Clasificador de imágenes</h1>
      <p class="subtitle">ResNet18 · CIFAR-10</p>
      <form id="uploadForm">
        <div class="dropzone" id="dropzone">
          <input type="file" name="file" id="fileInput" accept="image/*" required />
          <div class="dropzone-icon">📷</div>
          <div class="dropzone-text">Arrastra una imagen aquí o haz clic</div>
          <div class="dropzone-hint">JPEG, PNG o GIF</div>
          <div class="file-name" id="fileName"></div>
        </div>
        <button type="submit" class="btn" id="submitBtn" disabled>Clasificar imagen</button>
      </form>
      <div class="result" id="result">
        <div class="result-title">Resultado</div>
        <div class="result-class" id="resultClass"></div>
        <div class="result-prob" id="resultProb"></div>
      </div>
    </div>
    <p class="footer">
      <a href="/docs">API Docs</a> · <a href="/">Inicio</a>
    </p>
  </div>
  <script>
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const fileName = document.getElementById('fileName');
    const submitBtn = document.getElementById('submitBtn');
    const form = document.getElementById('uploadForm');
    const result = document.getElementById('result');
    const resultClass = document.getElementById('resultClass');
    const resultProb = document.getElementById('resultProb');

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(ev => {
      dropzone.addEventListener(ev, e => {
        e.preventDefault();
        e.stopPropagation();
        if (ev === 'dragleave' || ev === 'drop') dropzone.classList.remove('dragover');
        else dropzone.classList.add('dragover');
        if (ev === 'drop' && e.dataTransfer.files.length) fileInput.files = e.dataTransfer.files;
      });
    });

    fileInput.addEventListener('change', () => {
      const f = fileInput.files[0];
      fileName.textContent = f ? f.name : '';
      submitBtn.disabled = !f;
    });

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      if (!fileInput.files.length) return;
      result.classList.remove('visible');
      submitBtn.disabled = true;
      submitBtn.innerHTML = '<span class="spinner"></span>Clasificando…';
      const fd = new FormData();
      fd.append('file', fileInput.files[0]);
      try {
        const res = await fetch('/predict', { method: 'POST', body: fd });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          result.classList.add('visible', 'error');
          resultClass.textContent = data.detail || 'Error ' + res.status;
          resultProb.textContent = '';
        } else {
          result.classList.remove('error');
          result.classList.add('visible');
          resultClass.textContent = data.class_name || '—';
          resultProb.textContent = data.probability != null
            ? 'Confianza: ' + (data.probability * 100).toFixed(1) + '%'
            : '';
        }
      } catch (err) {
        result.classList.add('visible', 'error');
        resultClass.textContent = 'Error de conexión';
        resultProb.textContent = err.message || '';
      }
      submitBtn.disabled = false;
      submitBtn.textContent = 'Clasificar imagen';
    });
  </script>
</body>
</html>
"""


@app.get("/upload", response_class=HTMLResponse)
async def upload_form() -> str:
    """
    Página HTML con formulario para subir imágenes y ver el resultado de la clasificación.
    Accesible en http://127.0.0.1:8000/upload
    """
    return UPLOAD_PAGE_HTML


@app.get("/health")
async def health():
    """
    Simple health check: tries to load the model once.
    If something is wrong with the weights file, this will raise an error.
    """
    load_model()
    return {"status": "ok", "model_path": str(settings.model_path)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept an image upload (multipart/form-data) and return the predicted CIFAR-10 class.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file.")

    try:
        result = predict_image_bytes(image_bytes)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Error during prediction: {exc}") from exc

    return {
        "filename": file.filename,
        "class_index": result["class_index"],
        "class_name": result["class_name"],
        "probability": result["probability"],
    }


def main():
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.uvicorn_host,
        port=settings.uvicorn_port,
        reload=False,
    )


if __name__ == "__main__":
    main()

