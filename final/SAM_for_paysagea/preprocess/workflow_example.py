#!/usr/bin/env python3
"""
workflow_example.py
===================
Exemple complet d'un workflow utilisant preprocess_image.py
dans un contexte réel : détection SAM2 + annotation sur image originale
"""

import json
from pathlib import Path

import cv2
import numpy as np

from preprocess_image import (
    preprocess_image,
    save_metadata,
    load_metadata,
    convert_coordinates_to_original,
    convert_coordinates_to_resized
)


def step1_preprocess(input_image: str, max_side: int = 1024):
    """
    ÉTAPE 1 : Prétraitement de l'image
    
    Cette étape crée LA source de vérité pour toutes les étapes suivantes.
    """
    print("=" * 70)
    print("ÉTAPE 1 : PRÉTRAITEMENT")
    print("=" * 70)
    
    output_image = "preprocessed.jpg"
    metadata_file = "preprocessed.json"
    
    # Prétraiter l'image
    metadata = preprocess_image(input_image, output_image, max_side=max_side)
    
    # Sauvegarder les métadonnées
    save_metadata(metadata, metadata_file)
    
    print(f"\n✅ Fichiers créés :")
    print(f"   - {output_image} (image pour SAM2)")
    print(f"   - {metadata_file} (LA source de vérité)")
    
    return output_image, metadata_file


def step2_detect_objects(preprocessed_image: str, metadata_file: str):
    """
    ÉTAPE 2 : Détection d'objets avec SAM2 (simulé ici)
    
    Cette étape utilise l'image preprocessée et connait les dimensions
    grâce aux métadonnées.
    """
    print("\n" + "=" * 70)
    print("ÉTAPE 2 : DÉTECTION D'OBJETS (SAM2 simulé)")
    print("=" * 70)
    
    # Charger les métadonnées pour info
    metadata = load_metadata(metadata_file)
    resized_w, resized_h = metadata["preprocess"]["resized_size"]
    
    print(f"\n📐 Dimensions de travail : {resized_w}x{resized_h}")
    
    # Charger l'image preprocessée
    img = cv2.imread(preprocessed_image)
    
    # SIMULATION : Génération de détections aléatoires
    # Dans un vrai cas, ce serait : detections = sam2_model.detect(img)
    print("🔍 Détection en cours...")
    
    # Simuler 3 détections (boîtes englobantes)
    detections = {
        "boxes": [
            {
                "id": 1,
                "label": "person",
                "confidence": 0.95,
                "bbox": [100, 50, 300, 400],  # [x1, y1, x2, y2]
                "note": "coordonnées sur image redimensionnée"
            },
            {
                "id": 2,
                "label": "car",
                "confidence": 0.88,
                "bbox": [400, 200, 700, 450],
                "note": "coordonnées sur image redimensionnée"
            },
            {
                "id": 3,
                "label": "dog",
                "confidence": 0.92,
                "bbox": [150, 300, 350, 500],
                "note": "coordonnées sur image redimensionnée"
            }
        ],
        "metadata": {
            "source_image": preprocessed_image,
            "model": "SAM2 (simulé)",
            "coordinate_space": "resized"
        }
    }
    
    # Sauvegarder les détections
    detections_file = "detections.json"
    with open(detections_file, "w") as f:
        json.dump(detections, f, indent=2)
    
    print(f"\n✅ Détections sauvegardées : {detections_file}")
    print(f"   - {len(detections['boxes'])} objets détectés")
    print(f"   - Coordonnées dans l'espace redimensionné")
    
    return detections_file


def step3_annotate_original(
    original_image: str,
    detections_file: str,
    metadata_file: str
):
    """
    ÉTAPE 3 : Annotation de l'image originale
    
    Cette étape utilise :
    - L'image ORIGINALE (pleine résolution)
    - Les détections (coordonnées redimensionnées)
    - Les métadonnées (pour convertir les coordonnées)
    """
    print("\n" + "=" * 70)
    print("ÉTAPE 3 : ANNOTATION DE L'IMAGE ORIGINALE")
    print("=" * 70)
    
    # Charger l'image ORIGINALE
    img_original = cv2.imread(original_image)
    orig_h, orig_w = img_original.shape[:2]
    print(f"\n📐 Dimensions originales : {orig_w}x{orig_h}")
    
    # Charger les métadonnées
    metadata = load_metadata(metadata_file)
    scale_factor = metadata["preprocess"]["scale_factor"]
    print(f"📏 Facteur d'échelle : {scale_factor:.4f}")
    
    # Charger les détections
    with open(detections_file) as f:
        detections = json.load(f)
    
    print(f"\n🎨 Annotation de {len(detections['boxes'])} objets...")
    
    # Annoter chaque détection
    for box in detections["boxes"]:
        x1, y1, x2, y2 = box["bbox"]
        label = box["label"]
        confidence = box["confidence"]
        
        # CONVERSION : coordonnées redimensionnées → originales
        x1_orig, y1_orig = convert_coordinates_to_original(x1, y1, metadata)
        x2_orig, y2_orig = convert_coordinates_to_original(x2, y2, metadata)
        
        print(f"   {label}: [{x1}, {y1}, {x2}, {y2}] → "
              f"[{x1_orig:.0f}, {y1_orig:.0f}, {x2_orig:.0f}, {y2_orig:.0f}]")
        
        # Dessiner la boîte
        cv2.rectangle(
            img_original,
            (int(x1_orig), int(y1_orig)),
            (int(x2_orig), int(y2_orig)),
            (0, 255, 0),  # Vert
            3  # Épaisseur
        )
        
        # Ajouter le label
        label_text = f"{label} {confidence:.2f}"
        cv2.putText(
            img_original,
            label_text,
            (int(x1_orig), int(y1_orig) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )
    
    # Sauvegarder l'image annotée
    output_file = "annotated.jpg"
    cv2.imwrite(output_file, img_original)
    
    print(f"\n✅ Image annotée sauvegardée : {output_file}")
    print(f"   - Annotations sur image ORIGINALE")
    print(f"   - Pleine résolution : {orig_w}x{orig_h}")
    
    return output_file


def step4_create_comparison(
    original_image: str,
    preprocessed_image: str,
    annotated_image: str
):
    """
    ÉTAPE 4 : Créer une image de comparaison
    
    Montre côte à côte :
    - L'image originale
    - L'image preprocessée (utilisée pour SAM2)
    - L'image annotée (résultat final)
    """
    print("\n" + "=" * 70)
    print("ÉTAPE 4 : CRÉATION DE LA COMPARAISON")
    print("=" * 70)
    
    # Charger les images
    img_orig = cv2.imread(original_image)
    img_prep = cv2.imread(preprocessed_image)
    img_anno = cv2.imread(annotated_image)
    
    # Redimensionner toutes les images à la même hauteur pour comparaison
    target_height = 400
    
    def resize_to_height(img, height):
        h, w = img.shape[:2]
        ratio = height / h
        new_w = int(w * ratio)
        return cv2.resize(img, (new_w, height))
    
    img_orig_resized = resize_to_height(img_orig, target_height)
    img_prep_resized = resize_to_height(img_prep, target_height)
    img_anno_resized = resize_to_height(img_anno, target_height)
    
    # Ajouter des labels
    def add_label(img, text):
        img_copy = img.copy()
        cv2.putText(
            img_copy,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 0),  # Cyan
            2
        )
        return img_copy
    
    img_orig_labeled = add_label(img_orig_resized, "1. ORIGINAL")
    img_prep_labeled = add_label(img_prep_resized, "2. PREPROCESSED")
    img_anno_labeled = add_label(img_anno_resized, "3. ANNOTATED")
    
    # Concaténer horizontalement
    comparison = np.hstack([
        img_orig_labeled,
        img_prep_labeled,
        img_anno_labeled
    ])
    
    # Sauvegarder
    output_file = "comparison.jpg"
    cv2.imwrite(output_file, comparison)
    
    print(f"\n✅ Comparaison créée : {output_file}")
    
    return output_file


def main():
    """
    Workflow complet : de l'image brute à l'image annotée.
    """
    print("\n" + "🎯" * 35)
    print("WORKFLOW COMPLET : PRÉTRAITEMENT + SAM2 + ANNOTATION")
    print("🎯" * 35)
    
    # Vérifier qu'une image d'entrée existe
    input_image = "input.jpg"
    
    if not Path(input_image).exists():
        print(f"\n❌ Erreur : {input_image} n'existe pas")
        print("\n📝 Pour tester ce workflow :")
        print("   1. Créez une image de test :")
        print("      python -c \"from PIL import Image; ")
        print("      img = Image.new('RGB', (1920, 1080), 'blue'); ")
        print("      img.save('input.jpg')\"")
        print("   2. Relancez ce script")
        return
    
    try:
        # ÉTAPE 1 : Prétraiter
        preprocessed_image, metadata_file = step1_preprocess(
            input_image,
            max_side=1024
        )
        
        # ÉTAPE 2 : Détecter (simulé)
        detections_file = step2_detect_objects(
            preprocessed_image,
            metadata_file
        )
        
        # ÉTAPE 3 : Annoter l'originale
        annotated_image = step3_annotate_original(
            input_image,
            detections_file,
            metadata_file
        )
        
        # ÉTAPE 4 : Créer une comparaison
        comparison_image = step4_create_comparison(
            input_image,
            preprocessed_image,
            annotated_image
        )
        
        # Résumé final
        print("\n" + "🎉" * 35)
        print("WORKFLOW TERMINÉ AVEC SUCCÈS !")
        print("🎉" * 35)
        
        print("\n📁 Fichiers créés :")
        print(f"   1. {preprocessed_image} - Image preprocessée (pour SAM2)")
        print(f"   2. {metadata_file} - Métadonnées (LA source de vérité)")
        print(f"   3. {detections_file} - Détections SAM2")
        print(f"   4. {annotated_image} - Image finale annotée")
        print(f"   5. {comparison_image} - Comparaison visuelle")
        
        print("\n✅ Avantages de cette approche :")
        print("   ✓ Une seule source de vérité (preprocessed.json)")
        print("   ✓ Conversion de coordonnées automatique")
        print("   ✓ Image finale en pleine résolution")
        print("   ✓ Traçabilité complète du workflow")
        
    except Exception as e:
        print(f"\n❌ Erreur : {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
