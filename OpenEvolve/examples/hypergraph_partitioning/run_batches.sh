#!/bin/bash
# 200 iterations in three batches: 60, rest 2 h, 70, rest 3 h, 70.
# Each batch resumes from the latest checkpoint (start_evolution.py does that itself).
D="$(cd "$(dirname "$0")" && pwd)"
RESTS=(7200 10800)
cd $D
b=0
for n in 60 70 70; do
  last=$(ls openevolve_output/checkpoints 2>/dev/null | sed 's/checkpoint_//' | sort -n | tail -1); last=${last:-0}
  b=$((b+1)); from=$((last+1)); to=$((last+n)); log=run_${from}-${to}.log
  echo "$(date '+%F %T') batch $b: iterations $from-$to -> $log"
  python3 start_evolution.py --iterations $n > $log 2>&1
  rc=$?
  now=$(ls openevolve_output/checkpoints 2>/dev/null | sed 's/checkpoint_//' | sort -n | tail -1)
  echo "$(date '+%F %T') batch $b ended (exit $rc): $(grep -cE 'Iteration [0-9]+: Program' $log) iterations, $(grep -c 'Iteration [0-9]* error' $log) errors, limit-hit lines: $(grep -ciE 'usage limit|hit your limit' $log), checkpoint_${now:-none}"
  [ $b -lt 3 ] && { r=${RESTS[$((b-1))]}; echo "$(date '+%F %T') resting $((r/3600)) h"; sleep $r; }
done
echo "$(date '+%F %T') ALL BATCHES DONE at checkpoint_$(ls openevolve_output/checkpoints | sed 's/checkpoint_//' | sort -n | tail -1)"
