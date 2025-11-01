# Sign Language Translation Model

Repositorio para entrenamiento y evaluación de modelos de traducción de lenguaje de señas usando keypoints. El modelo utiliza una arquitectura Transformer para traducir secuencias de keypoints a texto.

Link donde está el archivo keypoints.h5 y el dataset LSA-T: https://github.com/midusi/LSA-T

## Requisitos
- Archivo `keypoints_cleaned.h5` en el directorio raíz del proyecto (correr clean_keypoints.py con el archivo keypoints.h5 en el directorio raíz)
- Archivo `meta.csv` en el directorio raíz del proyecto

## Instalación

1. Crear entorno virtual:

```bash
python -m venv venv
source venv/bin/activate 
```

2. Instalar dependencias:

```bash
pip install -r requirements.txt
```

3. Asegurar que `keypoints_cleaned.h5` y `meta.csv` estén en el directorio raíz.

## Uso

### Pipeline completo

Para ejecutar entrenamiento, evaluación e inferencia en secuencia:

```bash
./run_full_pipeline.sh
```

El script ejecuta tres fases:
- Entrenamiento del modelo
- Evaluación en el conjunto de test
- Inferencia con datos de ejemplo

Los logs se guardan en `pipeline_logs/` con timestamps.

### Configuración

Los archivos de configuración están en `configs/`. El pipeline usa `lsa_t_config.yaml` por defecto.

Para usar otra configuración, edita `run_full_pipeline.sh` y cambia la variable `CONFIG_PATH`:

```bash
CONFIG_PATH="configs/baseline.yaml"
```

O ejecuta comandos individuales:

```bash
# Entrenar
python main.py --config configs/lsa_t_config.yaml --mode train

# Evaluar
python main.py --config configs/lsa_t_config.yaml --mode evaluate

# Inferencia
python main.py --config configs/lsa_t_config.yaml --mode inference --input demo_keypoints.npy
```

## Estructura del proyecto

- `models/`: Implementación del modelo Transformer
- `training/`: Scripts de entrenamiento
- `evaluation/`: Métricas de evaluación (BLEU, accuracy)
- `inference/`: Generación de traducciones
- `utils/`: Utilidades de datos y vocabulario
- `configs/`: Archivos de configuración YAML
- `checkpoints/`: Modelos guardados durante entrenamiento
- `results/`: Resultados de evaluación

## Salidas

- Checkpoints: `checkpoints/lsa_t/best_model.pt`
- Resultados de evaluación: `results/lsa_t/evaluation_results.yaml`
- Logs de entrenamiento: `lsa_t_training.log`
- Logs del pipeline: `pipeline_logs/`

## Notas

- El script `run_full_pipeline.sh` está configurado para usar el entorno virtual `tesis/bin/activate`. Si tu entorno tiene otro nombre, modifica la línea correspondiente en el script.
- El modelo requiere CUDA para entrenamiento eficiente. Para usar CPU, cambia `device: "cuda"` a `device: "cpu"` en el archivo de configuración.
- Los parámetros del modelo (capas, dimensiones, learning rate) se configuran en los archivos YAML dentro de `configs/`.

