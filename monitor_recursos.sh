#!/bin/bash
# Script Universal - Monitoreo EXCLUSIVO del proceso de SLAM

LOG_FILE="consumo_rover_slam.txt"
echo "Tiempo(s) | CPU_SLAM(%) | RAM_SLAM(MB) | GPU_SLAM(%) | VRAM_SLAM(MiB)" > $LOG_FILE

CONTADOR=0

while true; do
    # 💡 1. Buscamos el PID dinámicamente de cualquier ejecutable que use "euroc"
    SLAM_PID=$(pgrep -f "euroc" | head -n 1)

    # Si no encuentra el PID, significa que el SLAM terminó o no ha empezado
    if [ -z "$SLAM_PID" ]; then
        # Le damos 3 segundos de cortesía al arrancar por si el SLAM tarda en levantar
        if [ $CONTADOR -gt 3 ]; then
            echo "[MONITOR] El proceso SLAM ha finalizado. Cerrando el log."
            exit 0
        fi
        sleep 1
        CONTADOR=$((CONTADOR + 1))
        continue
    fi

    # 💡 2. Captura CPU y RAM EXCLUSIVA del proceso usando su PID
    # ps nos da el % de CPU y los kilobytes (RSS) que usa ese PID exacto
    CPU_SLAM=$(ps -p $SLAM_PID -o %cpu --no-headers | tr -d ' ')
    RAM_KB=$(ps -p $SLAM_PID -o rss --no-headers | tr -d ' ')
    RAM_SLAM=$(awk "BEGIN {print $RAM_KB/1024}") # Convierte Kilobytes a Megabytes

    # 💡 3. Captura GPU y VRAM EXCLUSIVA que consume ese PID en NVIDIA
    # Buscamos en nvidia-smi la línea que pertenece a nuestro SLAM_PID
    GPU_SLAM=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    VRAM_SLAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)

    # Guarda los datos limpios en el archivo de texto
    echo "$CONTADOR | $CPU_SLAM | $RAM_SLAM | $GPU_SLAM | $VRAM_SLAM" >> $LOG_FILE
    
    sleep 1
    CONTADOR=$((CONTADOR + 1))
done