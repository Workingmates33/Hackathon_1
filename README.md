# BENIN Fongbe Tonal ASR Challenge 2026

Projet de reconnaissance automatique de la parole (ASR) tonale pour la langue Fongbé (Bénin).

## Organisation de l'équipe (4 Pôles)
- **Personne 1 (Vous) — Data Engineering** : Exploration, nettoyage, standardisation audio et métadonnées.
- **Personne 2 — Machine Learning / ASR** : Architecture acoustique, fine-tuning et entraînement de modèles.
- **Personne 3 — NLP / Analyse Tonale** : Phonétique, gestion des tons, syllabification (`syllabify()`).
- **Personne 4 — Évaluation / Déploiement** : Calcul du métrique pondéré, optimisation CPU (< 1 Go, < 2 Go RAM).

## Structure du Répertoire
```
Hackathon_1/
├── data/                  # Données brutes et traitées (exclues de git)
│   ├── raw/               # 350h de données d'origine
│   ├── processed/         # Audios standardisés (16kHz mono)
│   └── manifests/         # Manifests JSON/CSV pour l'entraînement
├── notebooks/             # Notebooks d'exploration et d'analyse
├── src/
│   ├── data/              # Scripts Data Engineering (Personne 1)
│   │   ├── audit.py       # Diagnostic des audios et textes
│   │   └── preprocess.py  # Standardisation
│   ├── models/            # Entraînement et architectures (Personne 2)
│   ├── nlp/               # Syllabification et traitement tonal (Personne 3)
│   └── eval/              # Métriques, bench CPU et inférence (Personne 4)
├── docs/
│   └── trello_planning.md # Tâches détaillées et plannings
├── .gitignore
└── README.md
```

## Règles de Confidentialité
Les données du challenge sont strictement confidentielles. Ne jamais committer de fichiers audio ou de transcriptions brutes sur le dépôt Git.
