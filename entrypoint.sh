#!/bin/bash
set -e

# Suppress noisy NumPy subnormal warnings at startup.
export PYTHONWARNINGS="${PYTHONWARNINGS:+$PYTHONWARNINGS,}ignore:The value of the smallest subnormal:UserWarning"

echo Preparing models...
python -m ai_video_analytics.prepare_models

echo Starting AI Video Analytics System using $NUM_WORKERS workers.

exec gunicorn --log-level $LOG_LEVEL\
     -w $NUM_WORKERS\
     -k uvicorn.workers.UvicornWorker\
     --keep-alive 60\
     --timeout 60\
     ai_video_analytics.api.main:app -b 0.0.0.0:18080
