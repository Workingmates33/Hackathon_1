# Documentation Technique — Pôle 1 : Data Engineering (Personne 1)
**Projet :** BENIN Fongbe Tonal ASR Challenge 2026  
**Branche Git :** `data-engineering`  
**Périmètre exclusif :** Personne 1 — Préparation, Audit et Compréhension des Données  

---

## 1. Traitement et Audit de `SampleSubmission.csv`

L'analyse automatisée réalisée par `src/data/submission_parser.py` sur le fichier `SampleSubmission.csv` met en évidence :
- **Nombre exact d'échantillons de test :** `9 407` lignes (`ID,Target`), aucun doublon, ordre strictement croissant.
- **Plage d'identifiants :** `test_000001` à `test_009438`.
- **Anomalie structurelle détectée pour protéger l'équipe :** **31 identifiants sont absents de la séquence numérique** :
  `test_000186`, `test_000638`, `test_000868`, `test_000884`, `test_001213`, `test_001334`, `test_001445`, `test_002749`, `test_002947`, `test_003086`, `test_003638`, `test_003956`, `test_004549`, `test_004571`, `test_004572`, `test_004573`, `test_005142`, `test_005217`, `test_005949`, `test_006330`, `test_006456`, `test_006732`, `test_006940`, `test_007155`, `test_007251`, `test_007335`, `test_007435`, `test_007723`, `test_008324`, `test_009185`, `test_009353`.
- **Action Data Engineering :** Génération automatique du fichier `data/manifests/test_manifest.jsonl` indexé exactement sur ces `9 407` identifiants officiels.

---

## 2. Couverture Complète des 6 Tâches de la Personne 1

1. **Explorer les fichiers audio et leurs transcriptions** (`notebooks/Pole1_Data_Engineering_Fongbe.ipynb` & `src/data/run_audit.py`).
2. **Identifier les formats, durées et caractéristiques des enregistrements** (`src/data/audit.py`) : taux d'échantillonnage, canaux, résolution en bits, durée min/médiane/max, énergie RMS, ratio de silence et détection d'écrêtage (*clipping*).
3. **Vérifier la qualité des annotations** (`src/data/submission_parser.py`) : séparation automatique des 3 sous-ensembles du challenge (`validated_multispeaker`, `biblical_reading`, `uncertain_tones`).
4. **Détecter les données manquantes ou problématiques** (`src/data/audit.py` & `src/data/fongbe_phonetics.py`) : rejet automatique des audios corrompus, silencieux, `< 0.5s` ou `> 20s`, des transcriptions vides, et détection/nettoyage des confusions Unicode (`ε` grec -> `ɛ` latin, numéros de versets bibliques).
5. **Préparer les données d'entraînement, de validation et les métadonnées** (`src/data/dataset.py`, `src/data/preprocess.py`, `src/data/augmentation.py`) : standardisation WAV `16 kHz Mono PCM 16-bit`, partitionnement **sans fuite de locuteur** (*Speaker-disjoint split*) et export de `train_manifest.jsonl`, `val_manifest.jsonl`, `test_manifest.jsonl` et `dataset_metadata_report.json`.
6. **Analyser les caractéristiques linguistiques du Fongbé** (`src/data/fongbe_phonetics.py`) : comptage des tons (haut `´`, bas `` ` ``, montant `ˇ`, descendant `ˆ`, moyen non marqué), caractères spéciaux (`ɖ`, `ɔ`, `ɛ`), digraphes (`gb`, `kp`, `hw`, `xw`, `ny`) et voyelles nasales (`an`, `ɛn`, `in`, `ɔn`, `un`).
