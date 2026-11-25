# Planification Trello & Roadmap - BENIN Fongbe Tonal ASR Challenge

## Vue d'ensemble du rôle : Personne 1 — Data Engineering

En tant que **Personne 1**, vous êtes le garant de la fondation de tout le projet. La qualité de vos données détermine directement le plafond de performance des modèles ASR des personnes 2, 3 et 4.

---

## Tableau Trello Recommandé

### Colonne 1 : 📋 Backlog & Cadrage
- [ ] **[P1-01] Structuration de l'environnement de données local**
  - Mettre en place les dossiers sécurisés `data/raw`, `data/interim`, `data/processed`.
  - Vérifier la règle de non-diffusion et configurer `.gitignore` (déjà fait).
- [ ] **[P1-02] Définition des formats d'échange (Manifests)**
  - Valider avec Personne 2 (ASR) le format d'entrée souhaité (JSON Lines ou CSV : `audio_filepath`, `duration`, `text_raw`, `speaker_id`, `subset`).
- [ ] **[P1-03] Inventaire des 350h de données**
  - Lister les sous-ensembles : enregistrements validés multi-locuteurs, annotations incertaines, lectures bibliques.

### Colonne 2 : 🔍 Exploration & Diagnostic (24–25 Nov)
- [ ] **[P1-04] Audit technique des fichiers audio**
  - Mesurer : taux d'échantillonnage (16kHz, 22kHz, 44.1kHz ?), formats (WAV, MP3 ?), mono/stéréo.
  - Détecter les audios corrompus ou illisibles.
  - Distribution des durées : audios trop courts (<0.5s) ou trop longs (>20s à tronquer/segmenter).
- [ ] **[P1-05] Audit textuel & analyse linguistique préliminaire**
  - Extraction de l'alphabet Fongbé présent dans les transcriptions (caractères spéciaux : ɖ, ɔ, ɛ, etc.).
  - Inventaire des diacritiques tonals (tons haut, bas, montant, descendant).
  - Détection des caractères polluants (HTML, ponctuations parasites, fautes de frappe).
- [ ] **[P1-06] Analyse de la fiabilité des annotations**
  - Isoler les transcriptions incertaines pour tester plus tard un pseudo-labeling ou un filtrage.

### Colonne 3 : ⚙️ Pipeline de Préparation & Nettoyage (26–29 Nov)
- [ ] **[P1-07] Script de standardisation audio**
  - Normalisation : conversion 16 kHz, mono, 16-bit PCM.
  - Normalisation du volume audio (silence removal ou VAD léger si nécessaire).
- [ ] **[P1-08] Partitionnement équilibré (Train / Validation / Test local)**
  - Split sans fuite de locuteurs (*speaker-disjoint split*) si les `speaker_id` sont disponibles.
  - Conserver un jeu de validation représentatif des accents et du Fongbé standard.
- [ ] **[P1-09] Génération des Manifests finaux**
  - Production de `train_manifest.jsonl`, `val_manifest.jsonl`.
  - Intégration préliminaire avec Personne 3 pour valider la compatibilité avec `syllabify()`.

### Colonne 4 : 🤝 Support Modélisation & Augmentation (30 Nov–5 Déc)
- [ ] **[P1-10] Stratégies de Data Augmentation audio**
  - Test de SpecAugment, ajout de bruit de fond léger, variation de vitesse (0.9x, 1.1x).
- [ ] **[P1-11] Traitement des 350h "bruyantes" / bibliques**
  - Proposition de stratégies d'entraînement progressif (ex: pré-entraînement sur tout le volume, fine-tuning sur données validées).

---

## Livrables clés pour Personne 1
1. `data_manifest.jsonl` : catalogue complet et propre des données.
2. `audit_report.md` : rapport complet (heures totales, locuteurs, distributions, métriques d'audio).
3. Script de preprocessing automatique reproductible (`src/data/preprocess.py`).
