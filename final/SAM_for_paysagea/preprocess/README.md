# 🎯 preprocess_image.py - Une seule vérité pour l'image

## Philosophie

Ce module garantit qu'il existe **une seule source de vérité** pour les dimensions et coordonnées d'image à travers toute votre pipeline.

### Le problème qu'il résout

❌ **Avant** : Chaos de coordonnées
```python
# Chaque script redimensionne à sa façon
sam2.py: resize to 1024
draw_boxes.py: resize to 800
annotate.py: resize to 1024 again

# Résultat : coordonnées incompatibles entre scripts
```

✅ **Après** : Une seule vérité
```python
# Un seul prétraitement
preprocess_image.py → photo_preprocessed.json

# Tous les scripts utilisent les mêmes métadonnées
sam2.py: lit photo_preprocessed.json
draw_boxes.py: lit photo_preprocessed.json
annotate.py: lit photo_preprocessed.json

# Résultat : coordonnées cohérentes partout
```

---

## 🎯 Responsabilités (et uniquement ça)

### 1️⃣ Charger l'image proprement
- ✅ Corriger l'orientation EXIF
- ✅ Forcer RGB (convertir RGBA, L, etc.)

### 2️⃣ Redimensionner intelligemment
- ✅ Garder le ratio d'aspect
- ✅ Limiter au `max_side` (ex: 1024px)
- ✅ **Pas d'upscale** (ne jamais agrandir)

### 3️⃣ Normaliser l'espace de coordonnées
- ✅ Conserver taille originale
- ✅ Conserver taille redimensionnée
- ✅ Calculer facteur d'échelle exact

### 4️⃣ Sauvegarder l'image preprocessée
- ✅ Format JPG optimisé (qualité 95)
- ✅ Nom explicite (ex: `photo_preprocessed.jpg`)

### 5️⃣ Retourner métadonnées JSON
- ✅ Format standardisé
- ✅ Prêt à être réutilisé partout

---

## 📄 Format du contrat JSON

```json
{
  "image_id": "sha256:a3f2e8b1c9d4f5e6",
  "source_filename": "IMG_5177.jpg",
  "preprocessed_filename": "IMG_5177_preprocessed.jpg",
  "preprocess": {
    "original_size": [1920, 1080],
    "resized_size": [1024, 576],
    "scale_factor": 0.5333,
    "max_side": 1024,
    "keep_ratio": true,
    "orientation": {
      "exif_present": true,
      "exif_orientation": 6,
      "applied_rotation_deg": 270
    }
  }
}
```

**Si pas d'EXIF :**

```json
{
  "image_id": "sha256:b7c3d2a1e4f8b9d0",
  "source_filename": "photo.jpg",
  "preprocessed_filename": "photo_preprocessed.jpg",
  "preprocess": {
    "original_size": [1920, 1080],
    "resized_size": [1024, 576],
    "scale_factor": 0.5333,
    "max_side": 1024,
    "keep_ratio": true,
    "orientation": {
      "exif_present": false,
      "exif_orientation": null,
      "applied_rotation_deg": 0
    }
  }
}
```

### Signification des champs

| Champ | Description | Exemple |
|-------|-------------|---------|
| `image_id` | Hash SHA-256 de l'image preprocessée (identifiant unique stable) | `sha256:a3f2e8b1c9d4f5e6` |
| `source_filename` | Nom du fichier source original | `IMG_5177.jpg` |
| `preprocessed_filename` | Nom du fichier preprocessé | `IMG_5177_preprocessed.jpg` |
| `original_size` | Dimensions de l'image source | `[1920, 1080]` |
| `resized_size` | Dimensions après redimensionnement | `[1024, 576]` |
| `scale_factor` | Facteur multiplicateur (resized/original) | `0.5333` |
| `max_side` | Limite imposée au côté le plus long | `1024` |
| `keep_ratio` | Ratio d'aspect préservé | `true` |
| `orientation.exif_present` | Tag EXIF orientation trouvé | `true` / `false` |
| `orientation.exif_orientation` | Valeur brute du tag EXIF (1-8) | `6` ou `null` |
| `orientation.applied_rotation_deg` | Rotation appliquée en degrés | `0`, `90`, `180`, `270` |

### 🔑 Pourquoi un `image_id` ?

L'`image_id` est un hash SHA-256 calculé sur le **contenu de l'image preprocessée**. C'est extrêmement utile pour :

1. **Identifier de manière unique** une image preprocessée
   ```python
   # Deux preprocessing du même fichier donnent le même image_id
   metadata1 = preprocess_image("photo.jpg", "out1.jpg", max_side=1024)
   metadata2 = preprocess_image("photo.jpg", "out2.jpg", max_side=1024)
   assert metadata1["image_id"] == metadata2["image_id"]  # ✅ True
   ```

2. **Détecter les modifications**
   ```python
   # Si l'image source change, l'image_id change aussi
   metadata_v1 = preprocess_image("photo.jpg", "out.jpg")
   # ... photo.jpg est modifiée ...
   metadata_v2 = preprocess_image("photo.jpg", "out.jpg")
   assert metadata_v1["image_id"] != metadata_v2["image_id"]  # ✅ True
   ```

3. **Fusionner des résultats de plusieurs pipelines**
   ```python
   # Pipeline 1 : SAM2 détection
   sam2_results = {
       "image_id": "sha256:a3f2e8b1c9d4f5e6",
       "boxes": [...]
   }
   
   # Pipeline 2 : Classification
   classifier_results = {
       "image_id": "sha256:a3f2e8b1c9d4f5e6",
       "labels": [...]
   }
   
   # Fusion sécurisée grâce à l'image_id
   if sam2_results["image_id"] == classifier_results["image_id"]:
       combined = merge(sam2_results, classifier_results)  # ✅ Safe
   ```

4. **Traçabilité dans des systèmes distribués**
   ```python
   # Serveur A prétraite
   metadata = preprocess_image("photo.jpg", "preprocessed.jpg")
   
   # Serveur B fait la détection (reçoit image_id via API)
   detections = {
       "image_id": metadata["image_id"],
       "timestamp": "2026-02-03T10:30:00Z",
       "results": [...]
   }
   
   # Serveur C agrège les résultats
   # → Peut vérifier que tous les résultats concernent bien la même image
   ```

### 📸 Comprendre l'orientation EXIF

Le champ `orientation` vous indique **précisément** ce qui s'est passé :

- **`exif_present: false`** → Aucune métadonnée EXIF, image intacte
- **`exif_present: true, applied_rotation_deg: 0`** → EXIF présent mais orientation normale (valeur 1)
- **`exif_present: true, applied_rotation_deg: 270`** → Image réellement pivotée de 270° (ex: photo prise en mode portrait)

| EXIF Value | Description | Rotation appliquée |
|------------|-------------|-------------------|
| 1 | Normal | 0° |
| 3 | Rotated 180° | 180° |
| 6 | Rotated 90° CW | 270° (pour redresser) |
| 8 | Rotated 270° CW | 90° (pour redresser) |

---

## 🚀 Utilisation

### CLI (ligne de commande)

```bash
# Usage basique
python preprocess_image.py photo.jpg photo_preprocessed.jpg

# Avec max_side personnalisé
python preprocess_image.py photo.jpg photo_preprocessed.jpg 2048

# Résultat :
# ✓ photo_preprocessed.jpg (image traitée)
# ✓ photo_preprocessed.json (métadonnées)
```

### Python (import)

```python
from preprocess_image import preprocess_image, load_metadata

# 1. Prétraiter
metadata = preprocess_image(
    input_path="photo.jpg",
    output_path="photo_preprocessed.jpg",
    max_side=1024
)

# 2. Réutiliser les métadonnées plus tard
metadata = load_metadata("photo_preprocessed.json")
print(metadata["preprocess"]["scale_factor"])  # 0.5333
```

---

## 🔄 Conversion de coordonnées

### De redimensionné → original

```python
from preprocess_image import convert_coordinates_to_original, load_metadata

metadata = load_metadata("photo_preprocessed.json")

# SAM2 détecte un point à (512, 288) sur l'image redimensionnée
x_resized, y_resized = 512, 288

# Convertir vers les coordonnées originales
x_orig, y_orig = convert_coordinates_to_original(
    x_resized, y_resized, metadata
)
# Résultat : (960.0, 540.0) sur l'image 1920×1080
```

### D'original → redimensionné

```python
from preprocess_image import convert_coordinates_to_resized

# Vous avez un point annoté manuellement à (1000, 600) sur l'originale
x_orig, y_orig = 1000, 600

# Convertir vers l'image redimensionnée (pour SAM2)
x_resized, y_resized = convert_coordinates_to_resized(
    x_orig, y_orig, metadata
)
# Résultat : (533.3, 320.0) sur l'image 1024×576
```

---

## 📁 Workflow recommandé

### Étape 1 : Prétraiter une fois

```bash
python preprocess_image.py photo.jpg photo_preprocessed.jpg 1024
```

**Crée :**
- `photo_preprocessed.jpg` → Image pour SAM2
- `photo_preprocessed.json` → **LA source de vérité** ⭐

### Étape 2 : Détecter avec SAM2

```python
# sam2_detect.py
import json
from preprocess_image import load_metadata

# Charger l'image preprocessée
img = cv2.imread("photo_preprocessed.jpg")

# Charger les métadonnées (pour référence)
metadata = load_metadata("photo_preprocessed.json")
print(f"Traitement d'une image {metadata['preprocess']['resized_size']}")

# Détecter avec SAM2
boxes = sam2_model.detect(img)

# Sauvegarder (coordonnées sur image redimensionnée)
with open("detections.json", "w") as f:
    json.dump({"boxes": boxes}, f)
```

### Étape 3 : Dessiner sur l'originale

```python
# draw_boxes.py
from preprocess_image import load_metadata, convert_coordinates_to_original

# Charger l'image ORIGINALE
img_original = cv2.imread("photo.jpg")

# Charger les métadonnées
metadata = load_metadata("photo_preprocessed.json")

# Charger les détections
with open("detections.json") as f:
    detections = json.load(f)

# Convertir et dessiner
for box in detections["boxes"]:
    x1, y1, x2, y2 = box
    
    # Convertir vers coordonnées originales
    x1_orig, y1_orig = convert_coordinates_to_original(x1, y1, metadata)
    x2_orig, y2_orig = convert_coordinates_to_original(x2, y2, metadata)
    
    # Dessiner sur l'image originale
    cv2.rectangle(img_original, 
                  (int(x1_orig), int(y1_orig)), 
                  (int(x2_orig), int(y2_orig)), 
                  (0, 255, 0), 2)

cv2.imwrite("photo_with_boxes.jpg", img_original)
```

---

## 📂 Structure de fichiers

```
project/
├── preprocess_image.py              # ⭐ Le module
├── photo.jpg                         # Image originale (intacte)
├── photo_preprocessed.jpg            # Image pour SAM2
├── photo_preprocessed.json           # LA source de vérité
├── detections.json                   # Résultats SAM2
└── photo_with_boxes.jpg              # Résultat final
```

---

## ✅ Avantages de cette approche

| Avantage | Description |
|----------|-------------|
| 🎯 **Une seule vérité** | Un seul fichier JSON fait autorité sur les dimensions |
| 🔄 **Conversion automatique** | Fonctions incluses pour convertir les coordonnées |
| 📏 **Précision garantie** | Pas d'arrondi flottant entre scripts |
| 🔍 **Traçabilité** | On sait toujours quelle image a été utilisée |
| 🧩 **Modularité** | Chaque script peut charger les mêmes métadonnées |
| 🚀 **Performance** | Redimensionnement une seule fois |

---

## ⚠️ Règles importantes

### ❌ Ne JAMAIS faire

```python
# ❌ Redimensionner dans chaque script
img = cv2.resize(img, (1024, 576))  # NON !

# ❌ Calculer le scale_factor à la main
scale = 1024 / 1920  # NON !

# ❌ Utiliser l'image preprocessée pour l'affichage final
cv2.imwrite("result.jpg", preprocessed_img)  # NON !
```

### ✅ Toujours faire

```python
# ✅ Utiliser l'image preprocessée pour SAM2
img = cv2.imread("photo_preprocessed.jpg")

# ✅ Charger les métadonnées
metadata = load_metadata("photo_preprocessed.json")

# ✅ Utiliser l'image originale pour le résultat final
img_original = cv2.imread("photo.jpg")
```

---

## 🔧 API de référence

### `preprocess_image(input_path, output_path, max_side=1024)`

Prétraite une image selon les règles définies.

**Paramètres :**
- `input_path` (str) : Chemin vers l'image source
- `output_path` (str) : Chemin de sortie pour l'image preprocessée
- `max_side` (int) : Taille max du côté le plus long (défaut: 1024)

**Retourne :**
- `dict` : Métadonnées de prétraitement

### `save_metadata(metadata, metadata_path)`

Sauvegarde les métadonnées dans un fichier JSON.

### `load_metadata(metadata_path)`

Charge les métadonnées depuis un fichier JSON.

**Retourne :**
- `dict` : Métadonnées

### `convert_coordinates_to_original(x, y, metadata)`

Convertit des coordonnées de l'image redimensionnée vers l'originale.

**Retourne :**
- `tuple[float, float]` : `(x_orig, y_orig)`

### `convert_coordinates_to_resized(x, y, metadata)`

Convertit des coordonnées de l'image originale vers la redimensionnée.

**Retourne :**
- `tuple[float, float]` : `(x_resized, y_resized)`

---

## 💡 Cas d'usage réels

### 1. Pipeline SAM2 + annotation

```bash
# 1. Prétraiter
python preprocess_image.py input.jpg input_preprocessed.jpg

# 2. Détecter avec SAM2 (utilise input_preprocessed.jpg)
python sam2_detect.py

# 3. Annoter l'originale (utilise input.jpg + métadonnées)
python draw_annotations.py
```

### 2. Traitement par lot

```python
from pathlib import Path
from preprocess_image import preprocess_image

for img_path in Path("images/").glob("*.jpg"):
    output = f"preprocessed/{img_path.stem}_preprocessed.jpg"
    preprocess_image(str(img_path), output, max_side=1024)
```

### 3. Web API

```python
from flask import Flask, request, jsonify
from preprocess_image import preprocess_image

app = Flask(__name__)

@app.route('/preprocess', methods=['POST'])
def api_preprocess():
    file = request.files['image']
    file.save('temp_input.jpg')
    
    metadata = preprocess_image(
        'temp_input.jpg',
        'temp_preprocessed.jpg',
        max_side=1024
    )
    
    return jsonify(metadata)
```

---

## 🧪 Tests

```python
# test_preprocess.py
from preprocess_image import preprocess_image, load_metadata
import json

# Test de base
metadata = preprocess_image("test.jpg", "test_preprocessed.jpg", 1024)

# Vérifications
assert metadata["preprocess"]["keep_ratio"] == True
assert metadata["preprocess"]["max_side"] == 1024
assert metadata["preprocess"]["scale_factor"] <= 1.0  # Pas d'upscale

# Test de cohérence
metadata_loaded = load_metadata("test_preprocessed.json")
assert metadata == metadata_loaded

print("✅ Tous les tests passent")
```

---

## 📝 Licence

Ce module est conçu pour être une pièce de base réutilisable dans vos pipelines de vision par ordinateur.

---

## 🤝 Contribution

Ce module doit rester **simple et focalisé**. Toute nouvelle fonctionnalité doit respecter le principe : **une seule responsabilité, une seule vérité**.

Si vous avez besoin de fonctionnalités supplémentaires (détection, annotation, etc.), créez des modules séparés qui **utilisent** `preprocess_image.py`, ne les ajoutez pas dedans.
