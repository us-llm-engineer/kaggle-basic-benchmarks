#!/usr/bin/env bash
# Best modelling so far, re-run on the dataset in ./data (cleaned) with the legacy scripts UNCHANGED. Steps run as independent processes; one log per step in logs/.
cd "$(dirname "$0")"; mkdir -p logs results; date -u +%FT%TZ > logs/_started; export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# wave 1: word-level members (cheap, parallel by fold) + char language model
for k in 0 1 2 3 4; do
  python3 train_glm.py $k 5 1 cont > logs/glm_f$k.log 2>&1 &
  python3 train_wordlm.py $k 4 1 > logs/wordlm_f$k.log 2>&1 &
  python3 train_fasttext.py $k word 30 0.1 20 > logs/fasttext_f$k.log 2>&1 &
done
wait; date -u +%FT%TZ > logs/_wave1_done
python3 train_charlm.py 9 > logs/charlm.log 2>&1 && python3 assemble_charlm.py > logs/assemble_charlm.log 2>&1; date -u +%FT%TZ > logs/_charlm_done
python3 -c "import spooky_features as F; from concurrent.futures import ProcessPoolExecutor; ex=ProcessPoolExecutor(5); list(ex.map(F.build_fold, range(5)))" > logs/features.log 2>&1
python3 -c "import train_nblr as T; T.C_GRID=[30]; T.main(1e9)" > logs/nblr.log 2>&1 &
python3 -c "import train_style as T; T.VARIANTS=['full']; T.main(1e9)" > logs/style.log 2>&1 &
CPU_FOLD_WORKERS=5 python3 train_char_svm.py > logs/char_svm.log 2>&1 &
wait; date -u +%FT%TZ > logs/_wave2_done
python3 assemble_glm.py > logs/assemble_glm.log 2>&1; python3 assemble_clean_pipeline.py > logs/assemble_members.log 2>&1
python3 stack_final.py nblr,style_gbdt,charlm,wordlm,glm,fasttext,char_svm > logs/stack_final.log 2>&1
for m in nblr style_gbdt charlm wordlm glm fasttext char_svm; do :; done; date -u +%FT%TZ > logs/_finished
