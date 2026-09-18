#!/usr/bin/env bash
# run_phaseB.sh — 본실험 페이즈 B(교란분리, Q2): 생성=EXAONE(논리검증=Qwen) × 동일 그리드 × 3런.
set -e
cd "$(dirname "$0")"
export LAW_OC="${LAW_OC:-lawbot2026injae}"
for r in 1 2 3; do
  for cond in rag norag; do
    /opt/conda/bin/python run_main.py --gen exaone --cond "$cond" --run "$r"
  done
done
/opt/conda/bin/python analyze_main.py
echo "PHASE_B_DONE"
