# 🚀 Guide du Pipeline Automatique

## Vue d'ensemble

Le script `pipeline_auto.py` automatise **tout le pipeline** de A à Z :
1. ✅ Preprocessing de l'image
2. ✅ Segmentation SAM automatique
3. ✅ Génération de 5 visualisations
4. ✅ Résumé complet (JSON + TXT)

**Un seul script → tout est fait automatiquement ! 🎯**

---

## 🎯 Utilisation simple

### Commande de base
```bash
python pipeline_auto.py photo.jpg
```

C'est tout ! Le script fait le reste automatiquement.

---

## 📊 Ce que fait le script

### ÉTAPE 1/4 : PREPROCESSING
- Charge l'image
- Corrige l'orientation EXIF
- Redimensionne à 25% (configurable)
- Calcule l'image_id (hash SHA256)
- Génère `*_preprocessed.json`

### ÉTAPE 2/4 : SEGMENTATION SAM
- Lit le `preprocess.json`
- Vérifie les dimensions
- Lance SAM en mode automatique
- Détecte tous les objets
- Génère `*_sam_output.json`

### ÉTAPE 3/4 : VISUALISATIONS
Crée **5 images** :
1. `01_preprocessed.png` - Image preprocessed seule
2. `02_all_segments.png` - Tous les segments colorés
3. `03_segments_with_ids.png` - Segments avec leurs IDs
4. `04_bounding_boxes.png` - Top 10 objets avec bounding boxes
5. `05_comparison.png` - Avant/Après côte à côte

### ÉTAPE 4/4 : RÉSUMÉ
Génère **2 fichiers de résumé** :
- `*_SUMMARY.json` - Résumé structuré
- `*_SUMMARY.txt` - Résumé lisible

---

## 📁 Structure des fichiers générés

```
pipeline_results/
├── IMG_5177-535x356.jpg                      # Image preprocessed
├── IMG_5177-535x356_preprocessed.json        # Métadonnées preprocess
├── IMG_5177-535x356_sam_output.json          # Résultats SAM
├── IMG_5177-535x356_01_preprocessed.png      # Viz 1: Image seule
├── IMG_5177-535x356_02_all_segments.png      # Viz 2: Tous segments
├── IMG_5177-535x356_03_segments_with_ids.png # Viz 3: Avec IDs
├── IMG_5177-535x356_04_bounding_boxes.png    # Viz 4: Bounding boxes
├── IMG_5177-535x356_05_comparison.png        # Viz 5: Comparaison
├── IMG_5177-535x356_SUMMARY.json             # Résumé JSON
└── IMG_5177-535x356_SUMMARY.txt              # Résumé texte
```

**10 fichiers générés automatiquement ! 🎉**

---

## 🎨 Exemples d'utilisation

### 1. Traitement simple
```bash
python pipeline_auto.py ma_photo.jpg
```

**Output console :**
```
============================================================
🚀 DÉMARRAGE DU PIPELINE AUTOMATIQUE
============================================================
Image : ma_photo.jpg
Output : pipeline_results

============================================================
ÉTAPE 1/4 : PREPROCESSING
============================================================
✓ Image chargée
  - path: ma_photo.jpg
  - size: 2140x1424
✓ Orientation corrigée
  - exif_orientation: 1
  - rotation: 0
✓ Image redimensionnée
  - original: 2140x1424
  - resized: 535x356
  - scale: 0.25
  - orientation: landscape
✓ Métadonnées sauvegardées
  - json: ma_photo-535x356_preprocessed.json
  - image_id: sha256:a319a26478988116

============================================================
ÉTAPE 2/4 : SEGMENTATION SAM
============================================================
✓ Preprocess chargé
  - image_id: sha256:a319a26478988116
  - size: [535, 356]
✓ Dimensions vérifiées
  - size: 535x356
✓ Modèle SAM chargé
  - device: cuda
  - model: vit_b
✓ Segmentation terminée
  - num_segments: 23
  - inference_time: 3.45s
✓ Résultats SAM sauvegardés
  - json: ma_photo-535x356_sam_output.json
  - size: 52.34 KB

============================================================
ÉTAPE 3/4 : VISUALISATIONS
============================================================
✓ Visualisation preprocessed créée
✓ Visualisation tous segments créée
  - segments: 23
✓ Visualisation avec IDs créée
✓ Visualisation bounding boxes créée
✓ Visualisation comparaison créée

============================================================
ÉTAPE 4/4 : RÉSUMÉ FINAL
============================================================
✓ Résumé JSON créé
  - path: ma_photo-535x356_SUMMARY.json
✓ Résumé texte créé
  - path: ma_photo-535x356_SUMMARY.txt

============================================================
✅ PIPELINE TERMINÉ AVEC SUCCÈS
============================================================

⏱️  Temps total : 8.23s

📊 Résultats :
  • 23 objets détectés
  • Temps SAM : 3.45s
  • 5 visualisations créées

🏆 Top 3 objets :
  1. Segment #0 : 18.42%
  2. Segment #1 : 12.56%
  3. Segment #2 : 8.73%

📁 Tous les fichiers dans : pipeline_results/
============================================================
```

### 2. Spécifier un dossier de sortie
```bash
python pipeline_auto.py photo.jpg --output mes_resultats
```

Tous les fichiers seront dans `mes_resultats/` au lieu de `pipeline_results/`

### 3. Utiliser un checkpoint SAM custom
```bash
python pipeline_auto.py photo.jpg --checkpoint /path/to/sam_vit_h_4b8939.pth
```

### 4. Batch processing
```bash
# Traiter toutes les images d'un dossier
for img in images/*.jpg; do
    echo "Traitement de $img..."
    python pipeline_auto.py "$img" --output "results/$(basename $img .jpg)"
done
```

---

## 📄 Exemple de résumé texte généré

```
============================================================
RÉSUMÉ DU PIPELINE - PREPROCESS → SAM
============================================================

📅 Date : 2025-02-10T15:23:45.123456
⏱️  Temps total : 8.23s

📸 IMAGE ORIGINALE
  • Fichier : IMG_5177.HEIC
  • Taille : 2140x1424

🔄 PREPROCESSING
  • Taille finale : 535x356
  • Scale : 0.25
  • Orientation : landscape

🎯 SEGMENTATION SAM
  • Segments détectés : 23
  • Temps d'inférence : 3.45s
  • Format : rle

📊 STATISTIQUES
  • Aire moyenne : 4.35%
  • Aire médiane : 2.18%
  • Plus petit : 0.12%
  • Plus grand : 18.42%

📈 DISTRIBUTION PAR TAILLE
  • tiny (<1%) : 8 objets
  • small (1-5%) : 10 objets
  • medium (5-20%) : 4 objets
  • large (>20%) : 1 objets

🏆 TOP 5 SEGMENTS
  1. Segment #0
     • Taille : 18.42%
     • Centre : (0.52, 0.74)
  2. Segment #1
     • Taille : 12.56%
     • Centre : (0.24, 0.41)
  3. Segment #2
     • Taille : 8.73%
     • Centre : (0.67, 0.33)
  4. Segment #3
     • Taille : 5.91%
     • Centre : (0.35, 0.58)
  5. Segment #4
     • Taille : 4.28%
     • Centre : (0.81, 0.62)

📁 FICHIERS GÉNÉRÉS
  • Preprocess : IMG_5177-535x356_preprocessed.json
  • SAM Output : IMG_5177-535x356_sam_output.json
  • Visualisations : 5
    - IMG_5177-535x356_01_preprocessed.png
    - IMG_5177-535x356_02_all_segments.png
    - IMG_5177-535x356_03_segments_with_ids.png
    - IMG_5177-535x356_04_bounding_boxes.png
    - IMG_5177-535x356_05_comparison.png
```

---

## 📊 Exemple de résumé JSON généré

```json
{
  "pipeline_info": {
    "date": "2025-02-10T15:23:45.123456",
    "total_time": "8.23s"
  },
  "input": {
    "original_file": "IMG_5177.HEIC",
    "original_size": [2140, 1424]
  },
  "preprocessing": {
    "resized_size": [535, 356],
    "scale": 0.25,
    "orientation": "landscape"
  },
  "segmentation": {
    "num_segments": 23,
    "inference_time": 3.45,
    "format": "rle"
  },
  "statistics": {
    "area_mean": 4.35,
    "area_median": 2.18,
    "area_min": 0.12,
    "area_max": 18.42,
    "distribution": {
      "tiny (<1%)": 8,
      "small (1-5%)": 10,
      "medium (5-20%)": 4,
      "large (>20%)": 1
    }
  },
  "top_segments": [
    {
      "id": 0,
      "area_percent": 18.42,
      "bbox": [0.42, 0.61, 0.31, 0.22],
      "centroid": [0.52, 0.74]
    },
    ...
  ],
  "outputs": {
    "preprocess_json": "IMG_5177-535x356_preprocessed.json",
    "sam_json": "IMG_5177-535x356_sam_output.json",
    "visualizations": [
      "IMG_5177-535x356_01_preprocessed.png",
      "IMG_5177-535x356_02_all_segments.png",
      "IMG_5177-535x356_03_segments_with_ids.png",
      "IMG_5177-535x356_04_bounding_boxes.png",
      "IMG_5177-535x356_05_comparison.png"
    ]
  }
}
```

---

## 🎯 Les 5 visualisations en détail

### 1. `01_preprocessed.png`
- Image preprocessed seule
- Permet de voir ce qui est traité par SAM

### 2. `02_all_segments.png`
- Tous les segments avec couleurs aléatoires
- Vue d'ensemble de tous les objets détectés
- Transparence à 40% pour voir l'image dessous

### 3. `03_segments_with_ids.png`
- Segments colorés avec IDs numérotés au centre
- Permet d'identifier chaque segment
- Utile pour référencer un segment spécifique

### 4. `04_bounding_boxes.png`
- Top 10 objets avec rectangles verts
- Label avec ID et pourcentage de taille
- Vue claire des plus gros objets

### 5. `05_comparison.png`
- Avant/Après côte à côte
- Gauche : image originale
- Droite : tous les segments colorés
- Parfait pour présenter les résultats

---

## 🔧 Options avancées

### Aide complète
```bash
python pipeline_auto.py --help
```

**Options disponibles :**
- `image` : Chemin de l'image (requis)
- `--output DIR` : Dossier de sortie (défaut: `pipeline_results`)
- `--checkpoint PATH` : Checkpoint SAM (défaut: `sam_vit_b_01ec64.pth`)

---

## 💡 Cas d'usage

### 1. Analyse rapide d'une image
```bash
python pipeline_auto.py test.jpg
# → Tout est généré en 1 commande
# → Voir pipeline_results/ pour les résultats
```

### 2. Documentation d'un dataset
```bash
for img in dataset/*.jpg; do
    python pipeline_auto.py "$img" --output "analysis/$(basename $img .jpg)"
done
# → Chaque image a son propre dossier d'analyse
```

### 3. Présentation client
```bash
python pipeline_auto.py client_image.jpg --output presentation
# → Montrer les 5 visualisations
# → Donner le SUMMARY.txt
```

### 4. Pipeline de production
```bash
# Intégrer dans un script plus large
python pipeline_auto.py input.jpg --output results
# Puis utiliser les JSON pour la suite du pipeline
python depth_estimation.py results/*_preprocessed.json
python fusion.py results/*_sam_output.json results/*_depth_output.json
```

---

## ⚠️ Prérequis

```bash
# Dépendances Python
pip install opencv-python pillow numpy matplotlib
pip install segment-anything pycocotools

# Checkpoint SAM vit_b
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

---

## 🐛 Résolution de problèmes

### Erreur : Image non trouvée
```bash
❌ Erreur : Image non trouvée : photo.jpg
```
**Solution :** Vérifier le chemin de l'image

### Erreur : Checkpoint SAM non trouvé
```bash
❌ Erreur : Checkpoint SAM non trouvé : sam_vit_b_01ec64.pth
```
**Solution :** Télécharger le checkpoint :
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

### CUDA out of memory
Le script détecte automatiquement et bascule sur CPU si nécessaire.

---

## 📈 Performance

**Temps moyens** (avec GPU CUDA) :
- Preprocessing : ~0.5s
- SAM (vit_b) : ~3-5s
- Visualisations : ~2-3s
- Total : **~6-9 secondes**

**Avec CPU :**
- Total : **~20-40 secondes**

---

## ✅ Avantages du pipeline automatique

| Avantage | Description |
|----------|-------------|
| **Simplicité** | 1 seule commande au lieu de 4 |
| **Cohérence** | Garantie que tout utilise le même preprocess |
| **Visualisations** | 5 images générées automatiquement |
| **Résumé** | Statistiques complètes en JSON + TXT |
| **Traçabilité** | image_id dans tous les fichiers |
| **Production-ready** | Gestion d'erreurs complète |

---

## 🎓 Comparaison : Manuel vs Automatique

### Méthode manuelle (4 commandes)
```bash
python preprocess_image.py photo.jpg
python sam_export_json_v2.py photo-535x356_preprocessed.json rle
python examples_json.py photo-535x356_sam_output.json photo-535x356.jpg
# Analyser manuellement les résultats...
```

### Méthode automatique (1 commande)
```bash
python pipeline_auto.py photo.jpg
# ✅ Tout est fait !
# ✅ Résumé complet généré
# ✅ 5 visualisations prêtes
```

**Gain de temps : ~80% ! 🚀**

---

## 🔮 Évolutions futures possibles

Le script est extensible pour ajouter :
- [ ] Support de multiples checkpoints SAM (vit_h, vit_l)
- [ ] Options de redimensionnement custom
- [ ] Intégration Depth dans le même pipeline
- [ ] Export PDF du résumé avec visualisations
- [ ] API REST pour traitement à distance
- [ ] Dashboard web interactif

---

**En résumé : Un script pour les gouverner tous ! 🎯**

```bash
python pipeline_auto.py votre_image.jpg
```
