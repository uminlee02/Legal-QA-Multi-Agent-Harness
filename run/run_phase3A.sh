#!/usr/bin/env bash
# run_phase3A.sh — 3번 페이즈A(주결과): target(형사47+민사29) × C0/C1/C2 × 2조건 × 2모델 × 3런.
set -e
cd "$(dirname "$0")"
export LAW_OC="${LAW_OC:-lawbot2026injae}"
for gen in qwen exaone; do
  for r in 1 2 3; do
    for cond in rag norag; do
      /opt/conda/bin/python run_main3.py --subset target --gen "$gen" --cond "$cond" --run "$r"
    done
  done
done
/opt/conda/bin/python analyze_main3.py
echo "PHASE_3A_DONE"
