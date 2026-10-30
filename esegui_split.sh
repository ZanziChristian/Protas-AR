#!/bin/bash
# Tutta la catena su uno split, un passo dopo l'altro (ogni passo già fatto si salta).
#
#   ./esegui_split.sh SPLIT [GPU]
#   nohup ./esegui_split.sh 1 0 > risultati/split1.log 2>&1 &      # in background, sulla GPU 0
#
# Tempi su una RTX A2000: segmentatori ~1 h, addestramento ~1 h 15, stime ~45 min.
set -e
cd "$(dirname "$0")"
PY=${PY:-python}
s=$1
[ -n "$2" ] && export CUDA_VISIBLE_DEVICES=$2
R=risultati/split$s
echo "split $s: inizio $(date '+%d/%m %H:%M')"
[ -f $R/token/codebook.npy ]                     || $PY -u tokenizzatore.py --split $s
[ -f $R/segmentazione/azioni_train.npz ]         || $PY -u segmenta.py --split $s
[ -f $R/ar_sf/modello_finale.pt ]                || $PY -u addestra.py --split $s
[ -f $R/stime_train.npz ]                        || $PY -u stima.py --split $s
$PY -u fusione.py --split $s
$PY -u fusione.py --split $s --calibra
$PY -u diagnosi.py --split $s
echo "split $s: fatto $(date '+%d/%m %H:%M')"
