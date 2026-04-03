"""
Exemple d'utilisation de preprocess_image.py
============================================
"""

from preprocess_image import (
    preprocess_image,
    save_metadata,
    load_metadata,
    convert_coordinates_to_original,
    convert_coordinates_to_resized
)


# 1️⃣ PRÉTRAITER UNE IMAGE
print("=" * 60)
print("ÉTAPE 1 : Prétraitement de l'image")
print("=" * 60)

metadata = preprocess_image(
    input_path="photo.jpg",
    output_path="photo_preprocessed.jpg",
    max_side=1024
)

# Sauvegarder les métadonnées
save_metadata(metadata, "photo_preprocessed.json")


# 2️⃣ RÉUTILISER LES MÉTADONNÉES DANS D'AUTRES SCRIPTS
print("\n" + "=" * 60)
print("ÉTAPE 2 : Réutilisation des métadonnées")
print("=" * 60)

# Dans un autre script (ex: sam2_detect.py, draw_boxes.py, etc.)
metadata = load_metadata("photo_preprocessed.json")

print("\n📊 Informations disponibles :")
print(f"   Taille originale : {metadata['preprocess']['original_size']}")
print(f"   Taille redimensionnée : {metadata['preprocess']['resized_size']}")
print(f"   Facteur d'échelle : {metadata['preprocess']['scale_factor']}")


# 3️⃣ CONVERSION DE COORDONNÉES
print("\n" + "=" * 60)
print("ÉTAPE 3 : Conversion de coordonnées")
print("=" * 60)

# Exemple : SAM2 détecte un objet à (512, 288) sur l'image redimensionnée
x_resized, y_resized = 512, 288
print(f"\n🎯 Point détecté sur image redimensionnée : ({x_resized}, {y_resized})")

# Convertir vers les coordonnées originales
x_orig, y_orig = convert_coordinates_to_original(x_resized, y_resized, metadata)
print(f"📍 Coordonnées dans image originale : ({x_orig:.1f}, {y_orig:.1f})")

# Et vice-versa
x_back, y_back = convert_coordinates_to_resized(x_orig, y_orig, metadata)
print(f"✓ Vérification (retour au redimensionné) : ({x_back:.1f}, {y_back:.1f})")


# 4️⃣ WORKFLOW COMPLET
print("\n" + "=" * 60)
print("WORKFLOW TYPIQUE")
print("=" * 60)

print("""
1. python preprocess_image.py photo.jpg photo_preprocessed.jpg 1024
   → Crée : photo_preprocessed.jpg + photo_preprocessed.json

2. python sam2_detect.py photo_preprocessed.jpg
   → Utilise : photo_preprocessed.jpg
   → Lit : photo_preprocessed.json (pour les dimensions)
   → Crée : detections.json (avec coordonnées sur image redimensionnée)

3. python draw_boxes.py photo.jpg detections.json photo_preprocessed.json
   → Utilise : photo.jpg (originale)
   → Lit : detections.json + photo_preprocessed.json
   → Convertit les coordonnées automatiquement
   → Crée : photo_with_boxes.jpg (sur l'image originale)

✅ Avantages :
   - Une seule source de vérité (photo_preprocessed.json)
   - Pas de confusion entre original et redimensionné
   - Conversion de coordonnées automatique
   - Traçabilité complète
""")


print("\n" + "=" * 60)
print("STRUCTURE DE FICHIERS RECOMMANDÉE")
print("=" * 60)

print("""
project/
├── preprocess_image.py          # Le script
├── photo.jpg                     # Image originale (intacte)
├── photo_preprocessed.jpg        # Image traitée (pour SAM2)
├── photo_preprocessed.json       # LA source de vérité ⭐
├── detections.json               # Résultats SAM2
└── photo_with_boxes.jpg          # Image finale annotée
""")
