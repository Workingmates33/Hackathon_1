# Ressources externes & Licences (Déclaration officielle)

> Rappel (`conventions.md` §5) : tout modèle pré-entraîné ou ressource publique doit être :
> 1. librement accessible,
> 2. **déclaré avant la fin de la compétition**,
> 3. compté dans la limite de **1 000 MB**,
> 4. sous une **licence autorisant l'usage commercial**.

## Modèles pré-entraînés évalués / utilisés

| Modèle | Source / URL | Licence | Usage commercial | Statut | Rôle |
| :--- | :--- | :--- | :---: | :--- | :--- |
| `facebook/w2v-bert-2.0` | https://huggingface.co/facebook/w2v-bert-2.0 | **MIT** | ✅ Oui | Candidat principal | Teacher (24L) et/ou Student élagué (12L/16L) + ONNX INT8 |
| `facebook/wav2vec2-xls-r-300m` | https://huggingface.co/facebook/wav2vec2-xls-r-300m | **Apache-2.0** | ✅ Oui | Backup | Alternative SSL CTC |
| `nvidia/stt_fr_fastconformer_ctc_large` | https://huggingface.co/nvidia/stt_fr_fastconformer_ctc_large | **CC-BY-4.0** | ✅ Oui | Candidat frugal | Alternative rapide mono-thread (~115M) |
| `facebook/mms-300m` / `mms-1b-all` | https://huggingface.co/facebook/mms-300m | CC-BY-NC-4.0 | ❌ **NON** | **INTERDIT** | Ne PAS utiliser |

## Données utilisées

- Données officielles uniquement : `Train_reliable.csv`, `Train_religious.csv`, `Train_uncertain.csv` (filtrage/pondération automatique uniquement).
- Aucune autre source biblique en Fongbé, aucune annotation manuelle, aucune donnée externe J'aime Ma Langue.

## Bibliothèques principales (inférence)

- `onnxruntime` (MIT)
- `numpy` (BSD-3-Clause)
- `soundfile` (BSD-3-Clause)
- `pyctcdecode` (Apache-2.0) + `kenlm` (LGPL-2.1) — *si modèle de langue retenu*
