#!/usr/bin/env bash
# run_phase3B.sh — 3번 페이즈B(대조군): control(기타+mixed 150) × 동일 그리드.
set -e
cd "$(dirname "$0")"
export LAW_OC="${LAW_OC:-lawbot2026injae}"
for gen in qwen exaone; do
  for r in 1 2 3; do
    for cond in rag norag; do
      /opt/conda/bin/python run_main3.py --subset control --gen "$gen" --cond "$cond" --run "$r"
    done
  done
done
/opt/conda/bin/python analyze_main3.py
echo "PHASE_3B_DONE"
