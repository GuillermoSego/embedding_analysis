#!/bin/bash

# --- Configuración ---
PORT=2284
LOCAL_BASE="/Users/guillermo_sego/Desktop/Maestría/Tesis/embeddings/sentiments"
REMOTE_USER="est_posgrado_guillermo.segura"
REMOTE_HOST="148.207.185.31"
REMOTE_BASE="/home/est_posgrado_guillermo.segura/EvoPrompt_Seg/data/cls"

# --- Lista de sentimientos ---
sentiments=("anger" "fear" "joy" "sadness" "surprise")

# --- Bucle principal ---
for sentiment in "${sentiments[@]}"; do
    echo "🔁 Copiando matriz S de: $sentiment ..."

    # Convertir solo la primera letra a mayúscula
    CAP_SENTIMENT="$(tr '[:lower:]' '[:upper:]' <<< "${sentiment:0:1}")${sentiment:1}"

    LOCAL_FILE="${LOCAL_BASE}/${sentiment}/matrix_S_${sentiment}.npy"
    REMOTE_DIR="${REMOTE_BASE}/semEval_${CAP_SENTIMENT}"

    # Copiar solo la matriz S
    scp -P "$PORT" "$LOCAL_FILE" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"

    echo "✅ Copiado: $LOCAL_FILE → ${REMOTE_DIR}/"
    echo ""
done

echo "🚀 Todas las matrices S fueron copiadas exitosamente."

