#!/bin/bash
# run_lead.sh LOGNAME zoom_lead.py-args... : thread-capped, niced run of zoom_lead.py; log in x_zoom/logs/LOGNAME.log
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=/home/liam/projects/vesuvius-work/x_zoom/logs; mkdir -p $L; name=$1; shift
exec nice -n 15 /home/liam/projects/vesuvius-work/envs/vcpy/bin/python "$(dirname "$0")/zoom_lead.py" "$@" > $L/$name.log 2>&1
