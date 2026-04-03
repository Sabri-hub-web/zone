# Guide d'utilisation de SAM avec le modèle vit_b

## 🚀 Installation rapide

```bash
# 1. Cloner le repository
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything

# 2. Installer SAM
pip install -e .

# 3. Installer les dépendances nécessaires
pip install opencv-python matplotlib pillow numpy
```

## 📥 Télécharger le modèle vit_b

```bash
# Télécharger le checkpoint vit_b (environ 375 MB)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

Ou avec Python :
```python
import urllib.request

url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
urllib.request.urlretrieve(url, "sam_vit_b_01ec64.pth")
```

## 💻 Exemples de code

### Exemple 1 : Segmentation automatique (tous les objets)

```python
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Charger l'image
image = cv2.imread('votre_image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Initialiser le modèle
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
sam.to(device="cuda")  # Ou "cpu" si pas de GPU

# Créer le générateur automatique
mask_generator = SamAutomaticMaskGenerator(sam)

# Générer tous les masques
masks = mask_generator.generate(image)

print(f"Nombre d'objets détectés : {len(masks)}")

# Visualiser les résultats
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    
    plt.figure(figsize=(15, 15))
    plt.imshow(image)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))
    
    plt.axis('off')
    plt.show()

show_anns(masks)
```

### Exemple 2 : Segmentation par point

```python
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

# Charger l'image
image = cv2.imread('votre_image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Initialiser le modèle avec le predictor
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
sam.to(device="cuda")  # Ou "cpu"

predictor = SamPredictor(sam)
predictor.set_image(image)

# Définir un point (x, y) sur l'objet à segmenter
input_point = np.array([[500, 375]])  # Coordonnées du point
input_label = np.array([1])  # 1 = point positif (sur l'objet)

# Prédire le masque
masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

# Afficher le meilleur masque (score le plus élevé)
best_mask_idx = np.argmax(scores)
best_mask = masks[best_mask_idx]

plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.imshow(best_mask, alpha=0.5, cmap='jet')
plt.plot(input_point[0, 0], input_point[0, 1], 'g*', markersize=20)
plt.title(f"Score: {scores[best_mask_idx]:.3f}")
plt.axis('off')
plt.show()
```

### Exemple 3 : Segmentation par boîte (bounding box)

```python
# Définir une bounding box [x_min, y_min, x_max, y_max]
input_box = np.array([425, 600, 700, 875])

# Prédire le masque
masks, scores, logits = predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_box[None, :],
    multimask_output=False,
)

# Visualiser
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.imshow(masks[0], alpha=0.5, cmap='jet')

# Dessiner la bounding box
x_min, y_min, x_max, y_max = input_box
rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                      fill=False, edgecolor='green', linewidth=3)
plt.gca().add_patch(rect)
plt.axis('off')
plt.show()
```

### Exemple 4 : Segmentation multiple avec points positifs et négatifs

```python
# Points positifs (sur l'objet) et négatifs (hors de l'objet)
input_points = np.array([
    [500, 375],  # Point positif
    [600, 400],  # Point positif
    [450, 300],  # Point négatif
])
input_labels = np.array([1, 1, 0])  # 1 = positif, 0 = négatif

masks, scores, logits = predictor.predict(
    point_coords=input_points,
    point_labels=input_labels,
    multimask_output=True,
)

# Afficher tous les masques proposés
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, (mask, score, ax) in enumerate(zip(masks, scores, axes)):
    ax.imshow(image)
    ax.imshow(mask, alpha=0.5, cmap='jet')
    
    # Afficher les points
    for point, label in zip(input_points, input_labels):
        color = 'green' if label == 1 else 'red'
        marker = '*' if label == 1 else 'x'
        ax.plot(point[0], point[1], color=color, marker=marker, 
                markersize=15, markeredgewidth=3)
    
    ax.set_title(f"Masque {idx+1} - Score: {score:.3f}")
    ax.axis('off')

plt.tight_layout()
plt.show()
```

## 🎯 Script complet prêt à l'emploi

```python
#!/usr/bin/env python3
"""
Script d'exemple pour utiliser SAM vit_b
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import sys

def load_model(checkpoint_path="sam_vit_b_01ec64.pth", device="cuda"):
    """Charge le modèle SAM vit_b"""
    print("Chargement du modèle...")
    sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    sam.to(device=device)
    print("✓ Modèle chargé avec succès")
    return sam

def segment_with_point(image_path, point_x, point_y, checkpoint="sam_vit_b_01ec64.pth"):
    """Segmente un objet à partir d'un point"""
    # Charger l'image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Charger le modèle
    sam = load_model(checkpoint)
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    
    # Prédire
    input_point = np.array([[point_x, point_y]])
    input_label = np.array([1])
    
    masks, scores, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    
    # Sauvegarder le meilleur masque
    best_idx = np.argmax(scores)
    output_mask = (masks[best_idx] * 255).astype(np.uint8)
    cv2.imwrite('output_mask.png', output_mask)
    print(f"✓ Masque sauvegardé : output_mask.png (score: {scores[best_idx]:.3f})")
    
    return masks, scores

def segment_all_objects(image_path, checkpoint="sam_vit_b_01ec64.pth"):
    """Segmente automatiquement tous les objets"""
    # Charger l'image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Charger le modèle
    sam = load_model(checkpoint)
    mask_generator = SamAutomaticMaskGenerator(sam)
    
    # Générer les masques
    print("Génération des masques...")
    masks = mask_generator.generate(image)
    print(f"✓ {len(masks)} objets détectés")
    
    return masks

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <image_path>")
        print("Exemple: python script.py photo.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Segmentation automatique
    masks = segment_all_objects(image_path)
    
    # Ou segmentation par point (décommenter si besoin)
    # masks, scores = segment_with_point(image_path, 500, 375)
```

## ⚙️ Options de configuration

### Optimiser les performances

```python
# Pour le générateur automatique
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,           # Nombre de points par côté (32 par défaut)
    pred_iou_thresh=0.86,         # Seuil de qualité (0.88 par défaut)
    stability_score_thresh=0.92,   # Seuil de stabilité (0.95 par défaut)
    crop_n_layers=0,              # Nombre de couches de crop (1 par défaut)
    crop_n_points_downscale_factor=1,  # Facteur de réduction
    min_mask_region_area=100,     # Aire minimale d'un masque
)
```

### Ajuster pour vitesse vs qualité

**Pour plus de vitesse** :
```python
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=16,  # Moins de points = plus rapide
    pred_iou_thresh=0.80,
    crop_n_layers=0,
)
```

**Pour plus de qualité** :
```python
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=64,  # Plus de points = plus précis
    pred_iou_thresh=0.90,
    crop_n_layers=1,
)
```

## 🐛 Résolution de problèmes

### Erreur CUDA out of memory
```python
# Utiliser le CPU au lieu du GPU
sam.to(device="cpu")
```

### Image trop grande
```python
# Redimensionner l'image avant traitement
import cv2
image = cv2.imread('image.jpg')
image = cv2.resize(image, (1024, 768))  # Ajuster selon vos besoins
```

### Modèle trop lent
- Utilisez un GPU si possible
- Réduisez `points_per_side`
- Redimensionnez vos images

## 📊 Performances attendues (vit_b)

- **Vitesse** : ~2-5 secondes par image (avec GPU)
- **Mémoire** : ~4 GB RAM (GPU)
- **Précision** : Bonne pour la plupart des applications
- **Taille du modèle** : 375 MB

## 🎓 Ressources supplémentaires

- [Repository GitHub officiel](https://github.com/facebookresearch/segment-anything)
- [Paper original](https://ai.facebook.com/research/publications/segment-anything/)
- [Démo en ligne](https://segment-anything.com/demo)
- [Dataset SA-1B](https://segment-anything.com/dataset/index.html)

## 💡 Conseils pratiques

1. **Pour débuter** : Commencez avec la segmentation automatique
2. **Pour la précision** : Utilisez les points positifs et négatifs
3. **Pour l'annotation** : Utilisez les bounding boxes
4. **Pour la production** : Optimisez les paramètres selon vos besoins
5. **Testez d'abord sur CPU** si vous n'avez pas de GPU, puis passez au GPU si nécessaire
