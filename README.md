# Sistema experimental de identificación de anemia

Aplicación web desarrollada con FastAPI.

## Pipeline

Imagen del ojo → U-Net → segmentación de conjuntiva →
YOLOv9-T → Grad-CAM++.

## Privacidad

La aplicación no persiste intencionalmente imágenes ni resultados.
El procesamiento se realiza temporalmente en memoria.

## Ejecutar en desarrollo

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
fastapi dev app/main.py