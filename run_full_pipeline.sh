#!/bin/bash

# Script para ejecutar pipeline completo: entrenamiento -> evaluación -> inferencia
# Uso: ./run_full_pipeline.sh
# Para tmux: tmux new-session -s pipeline './run_full_pipeline.sh'

set -e  # Salir si cualquier comando falla

# Configuración
CONFIG_PATH="configs/lsa_t_config.yaml"
LOG_DIR="pipeline_logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="$LOG_DIR/full_pipeline_$TIMESTAMP.log"

# Crear directorio de logs
mkdir -p "$LOG_DIR"

# Función para logging con timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$MAIN_LOG"
}

# Función para ejecutar comando y capturar salida
run_command() {
    local phase="$1"
    local cmd="$2"
    local log_file="$LOG_DIR/${phase}_$TIMESTAMP.log"
    
    log "INICIANDO FASE: $phase"
    log "Comando: $cmd"
    log "Log específico: $log_file"
    
    # Ejecutar comando y capturar tanto stdout como stderr
    if eval "$cmd" 2>&1 | tee "$log_file"; then
        log "FASE COMPLETADA: $phase"
        return 0
    else
        log "FASE FALLÓ: $phase"
        return 1
    fi
}

# Activar entorno virtual
log "Activando entorno virtual..."
source tesis/bin/activate || {
    log "Error activando entorno virtual 'tesis'"
    exit 1
}

# Cambiar al directorio del proyecto
cd /users/jbratti/keypoints-transformer-model-slt || {
    log "Error cambiando al directorio del proyecto"
    exit 1
}

log "INICIANDO PIPELINE COMPLETO DE SIGN LANGUAGE TRANSLATION"
log "============================================================="
log "Timestamp: $TIMESTAMP"
log "Directorio: $(pwd)"
log "Python: $(which python)"
log "Config: $CONFIG_PATH"
log "Log principal: $MAIN_LOG"
log "============================================================="

# Verificar archivos necesarios
log "Verificando archivos necesarios..."
if [[ ! -f "$CONFIG_PATH" ]]; then
    log "Config no encontrado: $CONFIG_PATH"
    exit 1
fi

if [[ ! -f "keypoints_cleaned.h5" ]]; then
    log "Dataset no encontrado: keypoints_cleaned.h5"
    exit 1
fi

if [[ ! -f "meta.csv" ]]; then
    log "Metadata no encontrada: meta.csv"
    exit 1
fi

log "Todos los archivos necesarios encontrados"

# FASE 1: ENTRENAMIENTO
log ""
log "=" * 60
if run_command "TRAINING" "python main.py --config $CONFIG_PATH --mode train"; then
    TRAINING_SUCCESS=true
    log "ENTRENAMIENTO COMPLETADO EXITOSAMENTE"
else
    TRAINING_SUCCESS=false
    log "ENTRENAMIENTO FALLÓ - Continuando con evaluación si existe modelo previo"
fi

# FASE 2: EVALUACIÓN
log ""
log "=" * 60
if run_command "EVALUATION" "python main.py --config $CONFIG_PATH --mode evaluate"; then
    EVALUATION_SUCCESS=true
    log "EVALUACIÓN COMPLETADA EXITOSAMENTE"
    
    # Mostrar resultados de evaluación
    if [[ -f "results/lsa_t/evaluation_results.yaml" ]]; then
        log "RESULTADOS DE EVALUACIÓN:"
        log "$(cat results/lsa_t/evaluation_results.yaml | sed 's/^/    /')"
    fi
else
    EVALUATION_SUCCESS=false
    log "EVALUACIÓN FALLÓ"
fi

# FASE 3: INFERENCIA (ejemplo con keypoints sintéticos)
log ""
log "=" * 60
log "Preparando datos de ejemplo para inferencia..."

# Crear keypoints sintéticos para demo de inferencia
python -c "
import numpy as np
import torch

# Crear keypoints sintéticos (ejemplo: 100 frames, 1086 features)
seq_len = 100
keypoint_dim = 1086
synthetic_keypoints = np.random.randn(seq_len, keypoint_dim).astype(np.float32)

# Guardar como .npy
np.save('demo_keypoints.npy', synthetic_keypoints)
print(f'Keypoints sintéticos creados: {synthetic_keypoints.shape}')
" 2>&1 | tee -a "$MAIN_LOG"

if [[ -f "demo_keypoints.npy" ]]; then
    if run_command "INFERENCE" "python main.py --config $CONFIG_PATH --mode inference --input demo_keypoints.npy"; then
        INFERENCE_SUCCESS=true
        log "INFERENCIA COMPLETADA EXITOSAMENTE"
    else
        INFERENCE_SUCCESS=false
        log "INFERENCIA FALLÓ"
    fi
else
    INFERENCE_SUCCESS=false
    log "No se pudieron crear keypoints sintéticos para inferencia"
fi

# RESUMEN FINAL
log ""
log "============================================================="
log "RESUMEN FINAL DEL PIPELINE"
log "============================================================="
log "Iniciado: $TIMESTAMP"
log "Finalizado: $(date '+%Y%m%d_%H%M%S')"
log ""

if [[ "$TRAINING_SUCCESS" == true ]]; then
    log "ENTRENAMIENTO: EXITOSO"
else
    log "ENTRENAMIENTO: FALLÓ"
fi

if [[ "$EVALUATION_SUCCESS" == true ]]; then
    log "EVALUACIÓN: EXITOSO"
else
    log "EVALUACIÓN: FALLÓ"
fi

if [[ "$INFERENCE_SUCCESS" == true ]]; then
    log "INFERENCIA: EXITOSO"
else
    log "INFERENCIA: FALLÓ"
fi

log ""
log "ARCHIVOS GENERADOS:"
log "  Log principal: $MAIN_LOG"
log "  Logs individuales: $LOG_DIR/"
if [[ -f "results/lsa_t/evaluation_results.yaml" ]]; then
    log "  Resultados evaluación: results/lsa_t/evaluation_results.yaml"
fi
if [[ -f "checkpoints/lsa_t/best_model.pt" ]]; then
    log "  Mejor modelo: checkpoints/lsa_t/best_model.pt"
fi

# Contar éxitos
SUCCESS_COUNT=0
[[ "$TRAINING_SUCCESS" == true ]] && ((SUCCESS_COUNT++))
[[ "$EVALUATION_SUCCESS" == true ]] && ((SUCCESS_COUNT++))
[[ "$INFERENCE_SUCCESS" == true ]] && ((SUCCESS_COUNT++))

log ""
if [[ $SUCCESS_COUNT -eq 3 ]]; then
    log "PIPELINE COMPLETADO: 3/3 FASES EXITOSAS"
    exit 0
elif [[ $SUCCESS_COUNT -eq 2 ]]; then
    log "PIPELINE PARCIALMENTE EXITOSO: 2/3 FASES EXITOSAS"
    exit 0
elif [[ $SUCCESS_COUNT -eq 1 ]]; then
    log "PIPELINE PARCIALMENTE EXITOSO: 1/3 FASES EXITOSAS"
    exit 1
else
    log "PIPELINE FALLÓ: 0/3 FASES EXITOSAS"
    exit 1
fi
