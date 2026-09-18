#!/usr/bin/env bash
# 실배포 구성(생성=EXAONE) KoBLEX 재측정 — 새 메인 표. 기존 Qwen 파일 불변(신규 --out만).
cd "$(dirname "$0")"
export LAW_OC="${LAW_OC:-lawbot2026injae}"
EXA="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"; U8001="http://localhost:8001/v1"
echo "###### [M1-1/4] EXAONE 3-way 맨몸+사후 (koblex_exaone_main) $(date +%H:%M:%S) ######"
MODEL="$EXA" VLLM_BASE_URL="$U8001" /opt/conda/bin/python run_koblex.py --out koblex_exaone_main
echo "###### [M1-2/4] EXAONE RAG (koblex_exaone_main_rag) $(date +%H:%M:%S) ######"
MODEL="$EXA" VLLM_BASE_URL="$U8001" /opt/conda/bin/python run_koblex.py --rag results/koblex_retrieval_k5.json --out koblex_exaone_main_rag
echo "###### [M2-3/4] self(EXAONE) vs cross(Qwen) RAG (crossverify_exaone_gen_n30) $(date +%H:%M:%S) ######"
/opt/conda/bin/python run_crossverify_exaone.py --n 30 --out crossverify_exaone_gen_n30
echo "###### [M2-4/4] self vs cross 맨몸 (crossverify_exaone_gen_norag_n30) $(date +%H:%M:%S) ######"
/opt/conda/bin/python run_crossverify_exaone.py --n 30 --no-rag --out crossverify_exaone_gen_norag_n30
echo "###### ALL DONE $(date +%H:%M:%S) ######"
