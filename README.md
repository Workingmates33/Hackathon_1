# 🇧🇯 BENIN Fongbe Tonal ASR Challenge 2026 — Branche `data-engineering` (Personne 1)

Ce dépôt (branche **`data-engineering`**) contient l'ensemble des travaux et livrables du **Pôle 1 — Data Engineering (Personne 1)** : exploration, audit du fichier `SampleSubmission.csv`, contrôle qualité acoustique et textuel, analyse linguistique du Fongbé et préparation des jeux de données propres pour le reste de l'équipe.

## Exécution Rapide (Pôle 1)

- **Notebook Jupyter interactif complet (pré-exécuté avec tableaux et graphiques)** :
  `notebooks/Pole1_Data_Engineering_Fongbe.ipynb`
- **Pipeline CLI d'audit et génération des manifests** :
  ```bash
  python3 -m src.data.run_audit
  ```
- **Tests unitaires Data Engineering** :
  ```bash
  python3 -m unittest discover tests/ -v
  ```

## Structure du Pôle 1 (Personne 1)
```
Hackathon_1/
├── data/                              # Ignoré par Git (données confidentielles)
│   ├── raw/SampleSubmission.csv       # 9 407 IDs officiels de test
│   ├── processed/                     # Audios standardisés WAV 16kHz mono PCM 16-bit
│   └── manifests/                     # Livrables Personne 1 : train/val/test.jsonl + rapport JSON
├── notebooks/
│   └── Pole1_Data_Engineering_Fongbe.ipynb  # Notebook complet de vos 6 missions Data Engineering
├── src/
│   └── data/
│       ├── config.py                  # Constantes audio (16 kHz, mono, 0.5s–20s)
│       ├── submission_parser.py       # Audit de SampleSubmission.csv & classification des sources
│       ├── audit.py                   # Détection d'anomalies audio/texte (silence, durée, glyphes)
│       ├── preprocess.py              # Conversion FFmpeg 16kHz mono 16-bit PCM
│       ├── fongbe_phonetics.py        # Analyse linguistique Fongbé (tons, nasales, digraphes, ɖ/ɔ/ɛ)
│       ├── syllabifier.py             # Aperçu syllabique de contrôle ("Un nɔ́ ɖu nǔ" -> "un nɔ1 ɖu nu3")
│       ├── augmentation.py            # Augmentation audio préservant les tons (SNR, gain, SpecAugment)
│       ├── dataset.py                 # Split sans fuite locuteur & génération des manifests JSONL
│       ├── synthetic_data.py          # Jeu d'essai Fongbé (cas sains + anomalies types)
│       └── run_audit.py               # Point d'entrée CLI Data Engineering
├── tests/
│   └── test_data_pipeline.py          # Suite de tests unitaires du Pôle 1
└── docs/
    ├── DOCUMENTATION.md               # Documentation technique complète Personne 1
    └── trello_planning.md             # Planification Trello
```
