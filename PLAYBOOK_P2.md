# Playbook Personne 2 (ML / ASR) — Frugal Tonal ASR for Fongbe

## 1. Générer les splits miroir du Test Set (local ou Kaggle)

```bash
python src/make_splits.py --data-root . --out-dir splits --val-book JDG
```
Produit :
- `splits/train_clean.csv` : **63 008 phrases (137.9 h)** (`Train_reliable` train + 8 livres `Train_religious`)
- `splits/val_general.csv` : **2 396 phrases (5.25 h)** (`recommended_validation == 1`, 14 voix inédites)
- `splits/val_religious.csv` : **1 194 phrases (2.81 h)** (livre `JDG` tenu de côté)
- `splits/val_mirror.csv` : combiné (**66.7 % général / 33.3 % religieux**, miroir du Test Set **67.8 % / 32.2 %**)
- `splits/vocab.json` : **37 tokens NFD** (29 lettres + 5 tons + `|` + `[UNK]` + `[PAD]`)

---

## 2. Benchmark frugalité CPU AVANT entraînement (Kaggle CPU / Linux / WSL2)

Installe sur Linux/Kaggle :
```bash
pip install -q "transformers==5.14.1" soundfile "rapidfuzz==3.14.5" onnx onnxruntime
```

Exporte `w2v-bert-2.0` en **24 couches**, **16 couches** et **12 couches** (INT8) et teste avec `measure_frugality.py` :
```bash
# Test 12 couches (~300M -> ~310 MB INT8)
python src/export_onnx_int8.py --dry-run-pretrained --num-layers 12 --vocab splits/vocab.json --out-dir weights/bench_12L
python measure_frugality.py \
    --weights weights/bench_12L \
    --command "python src/infer_onnx.py --weights weights/bench_12L --csv {csv} --root {root} --output {output}" \
    --csv splits/val_mirror.csv --root /tmp/fongbe --limit 100

# Répéter avec --num-layers 16 et --num-layers 24 pour choisir le plus gros modèle qui garde RTF <= 0.75 et RAM <= 1.5 GB !
```

---

## 3. Entraînement GPU (Kaggle T4 / Colab)

```bash
python src/train_ctc.py \
    --data-dir /tmp/fongbe \
    --train-csv splits/train_clean.csv \
    --val-csv splits/val_mirror.csv \
    --vocab splits/vocab.json \
    --out-dir outputs/w2vbert_12L \
    --num-layers 12 \
    --spec-augment \
    --max-steps 6000 \
    --eval-steps 500 \
    --batch-size 4 \
    --grad-accum 8
```
*(Si la session Kaggle coupe après 9–12 h, relance simplement avec `--resume-from outputs/w2vbert_12L/latest` !)*

---

## 4. Évaluation miroir complète + analyse des tons (pour Personne 3)

```bash
python src/evaluate.py \
    --reference splits/val_mirror.csv \
    --predictions outputs/val_pred.csv \
    --output outputs/eval_report.json \
    --show-errors 15
```

---

## 5. Export final INT8 + vérification frugalité + soumission Zindi

```bash
# 1. Export ONNX INT8 + encodeur isolé (vérification finale)
python src/export_onnx_int8.py \
    --model-dir outputs/w2vbert_12L/best \
    --out-dir weights/final_int8 \
    --export-isolated-encoder

# 2. Vérification officielle sur validation (Linux / Kaggle CPU)
python measure_frugality.py \
    --weights weights/final_int8 \
    --command "python src/infer_onnx.py --weights weights/final_int8 --csv {csv} --root {root} --output {output}" \
    --csv splits/val_mirror.csv --root /tmp/fongbe --limit 300 --report outputs/frugality_val.json

# 3. Inférence Test.csv sur le MÊME pipeline CPU 1-thread + syllabification Zindi
python src/infer_onnx.py \
    --weights weights/final_int8 \
    --csv Test.csv --root /tmp/fongbe \
    --output outputs/test_plain.csv \
    --syllabified-output outputs/submission.csv
```
