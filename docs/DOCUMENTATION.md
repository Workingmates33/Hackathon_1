# Documentation Technique — Pôle Data Engineering (Personne 1)
**Projet :** BENIN Fongbe Tonal ASR Challenge 2026  
**Branche Git :** `data-engineering`  
**Date :** 24 novembre 2026  

---

## 1. Contexte & Objectif

Dans le cadre du **BENIN Fongbe Tonal ASR Challenge**, la mission du pôle **Data Engineering (Personne 1)** est de fournir à l'équipe un pipeline robuste, propre et reproductible pour ingérer, auditer, standardiser et segmenter les **350 heures** de données audio en langue Fongbé.

Le Fongbé étant une **langue tonale** (le ton modifie le sens sémantique d'un mot), le traitement des données doit impérativement préserver l'information tonale tout en respectant les contraintes strictes de déploiement (CPU modeste, <= 2 Go de RAM, <= 1 Go pour le modèle).

---

## 2. Synthèse des Réalisations

### A. Gestion de Version & Sécurité des Données
- **Branche Git dédiée :** `data-engineering`.
- **Confidentialité (`.gitignore`) :** Exclusion absolue des fichiers audio bruts (`*.wav`, `*.mp3`, `*.flac`), archives (`.zip`, `.tar.gz`), checkpoints de modèles et du dossier `/data/` pour respecter les clauses de non-diffusion des organisateurs.

### B. Arborescence du Projet
```
Hackathon_1/
├── data/                      # Données (ignorées par Git)
│   ├── raw/                   # Audios et métadonnées brutes d'origine (350h)
│   ├── processed/             # Audios convertis (16kHz mono 16-bit PCM)
│   └── manifests/             # Manifests JSON Lines (train, val, test)
├── docs/
│   ├── DOCUMENTATION.md       # Présente documentation technique
│   └── trello_planning.md     # Cartes Trello et planning de l'équipe
├── src/
│   └── data/
│       ├── __init__.py        # Package data
│       ├── config.py          # Paramètres audio et contraintes matérielles
│       ├── fongbe_phonetics.py# Alphabet, voyelles nasales/orales et diacritiques
│       ├── syllabifier.py     # Fonction officielle de syllabification tonale
│       ├── preprocess.py      # Standardisation audio via FFmpeg
│       ├── dataset.py         # Construction des manifests et split sans fuite locuteur
│       ├── audit.py           # Fonctions d'inspection audio & intégrité
│       └── run_audit.py       # Point d'entrée CLI pour lancer les audits
├── tests/
│   └── test_data_pipeline.py  # Suite de tests unitaires automatisés
├── requirements.txt           # Dépendances Python du pôle
└── README.md                  # Présentation du projet
```

---

## 3. Détail des Modules Développés

### 1. Configuration & Contraintes (`src/data/config.py`)
Centralise les constantes du projet :
- Taux d'échantillonnage : `16 000 Hz`
- Canaux : `1` (Mono)
- Format d'encodage : `16-bit PCM`
- Plage de durée retenue : `0.5s` à `20.0s`
- Plafond mémoire inférence : `2 048 Mo`
- Plafond poids modèle : `1 000 Mo`

### 2. Phonétique & Linguistique Fongbé (`src/data/fongbe_phonetics.py`)
- Gestion des voyelles orales (`a, e, ɛ, i, o, ɔ, u`) et nasales (`an, ɛn, in, ɔn, un`).
- Prise en charge des caractères spécifiques de l'alphabet béninois (`ɖ`, `ɔ`, `ɛ`) et des labiovélaires (`gb`, `kp`, `hw`, `xw`, `ny`).
- Décomposition Unicode (NFD) pour isoler les diacritiques tonals des caractères de base.

### 3. Syllabification & Numérotation des Tons (`src/data/syllabifier.py`)
Conformité avec la métrique officielle du challenge (*Weighted Syllable Word Error Rate*) :
- Décomposition NFD et extraction des tons.
- Remplacement des diacritiques par des suffixes numériques :
  - Ton haut (´) -> `1`
  - Ton descendant (ˆ) -> `2`
  - Ton montant (ˇ) -> `3`
  - Ton bas (`) -> `4`
- **Exemple validé par test :**
  `"Un nɔ́ ɖu nǔ"` -> `"un nɔ1 ɖu nu3"`

### 4. Normalisation Audio (`src/data/preprocess.py`)
- Pipeline de conversion automatisé s'appuyant sur FFmpeg.
- Conversion standardisée en `pcm_s16le`, `16000 Hz`, mono.
- Calcul de la durée exacte pour chaque fichier via `ffprobe`.

### 5. Manifests & Partitionnement sans Fuite (`src/data/dataset.py`)
- Production de manifests sous format JSON Lines (`.jsonl`), le standard pour Wav2Vec2 / Whisper / NeMo.
- Format de chaque entrée :
  ```json
  {
    "audio_filepath": "data/processed/sample_001.wav",
    "duration": 3.45,
    "text_raw": "Un nɔ́ ɖu nǔ",
    "text_syllabified": "un nɔ1 ɖu nu3",
    "speaker_id": "spk_12"
  }
  ```
- **Stratégie de découpage (*Speaker-disjoint split*) :** Garantit qu'aucun locuteur présent dans l'ensemble de validation n'apparaît dans l'ensemble d'entraînement.

### 6. Outil CLI d'Audit (`src/data/run_audit.py`)
Permet d'exécuter l'inspection dès l'arrivée des données en ligne de commande :
```bash
python3 -m src.data.run_audit --raw-dir data/raw --processed-dir data/processed
```

---

## 4. Validation & Tests

La suite de tests unitaires `tests/test_data_pipeline.py` a été exécutée avec succès :
- `test_syllabification_baseline` : **PASS** (validation exacte de l'exemple du règlement)
- `test_fongbe_special_characters` : **PASS** (intégrité de `ɖ`, `ɔ`, `ɛ`)
- `test_manifest_entry` : **PASS** (structure du manifest JSONL)
- `test_speaker_disjoint_split` : **PASS** (intersection locuteurs Train / Val vide)

---

## 5. Prochaines Étapes pour Personne 1

1. Déposer les audios et métadonnées dans `data/raw/`.
2. Lancer la commande d'audit : `python3 -m src.data.run_audit`.
3. Générer les manifests finaux pour Personne 2 (ASR) et Personne 3 (NLP).
