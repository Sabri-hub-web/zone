#!/usr/bin/env python3
"""
Exemples pratiques d'utilisation des exports JSON de SAM
"""

import json
import cv2
import numpy as np
from pycocotools import mask as mask_utils
import matplotlib.pyplot as plt

def load_sam_json(json_path):
    """Charge un fichier JSON SAM"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['sam_output']

def example_1_count_objects(json_path):
    """Exemple 1 : Compter les objets"""
    print("\n=== EXEMPLE 1 : Compter les objets ===")
    
    sam_data = load_sam_json(json_path)
    
    print(f"Nombre total d'objets : {sam_data['num_segments']}")
    print(f"Taille de l'image : {sam_data['image_size']}")
    print(f"Format des masques : {sam_data['format']}")

def example_2_largest_objects(json_path, top_n=5):
    """Exemple 2 : Trouver les N plus gros objets"""
    print(f"\n=== EXEMPLE 2 : Top {top_n} objets par taille ===")
    
    sam_data = load_sam_json(json_path)
    segments = sam_data['segments'][:top_n]
    
    for i, seg in enumerate(segments, 1):
        area_percent = seg['area_ratio'] * 100
        print(f"{i}. Segment ID {seg['segment_id']}: {area_percent:.2f}% de l'image")
        print(f"   Bbox: {seg['bbox']}")
        print(f"   Centre: {seg['centroid']}")

def example_3_filter_by_size(json_path, min_area=0.01, max_area=0.5):
    """Exemple 3 : Filtrer par taille"""
    print(f"\n=== EXEMPLE 3 : Objets entre {min_area*100}% et {max_area*100}% ===")
    
    sam_data = load_sam_json(json_path)
    
    filtered = [
        seg for seg in sam_data['segments']
        if min_area <= seg['area_ratio'] <= max_area
    ]
    
    print(f"Nombre d'objets trouvés : {len(filtered)}")
    for seg in filtered:
        print(f"  - ID {seg['segment_id']}: {seg['area_ratio']*100:.2f}%")

def example_4_filter_by_position(json_path, region="left"):
    """Exemple 4 : Filtrer par position"""
    print(f"\n=== EXEMPLE 4 : Objets dans la région '{region}' ===")
    
    sam_data = load_sam_json(json_path)
    
    filtered = []
    for seg in sam_data['segments']:
        cx, cy = seg['centroid']
        
        if region == "left" and cx < 0.5:
            filtered.append(seg)
        elif region == "right" and cx >= 0.5:
            filtered.append(seg)
        elif region == "top" and cy < 0.5:
            filtered.append(seg)
        elif region == "bottom" and cy >= 0.5:
            filtered.append(seg)
        elif region == "center" and 0.25 < cx < 0.75 and 0.25 < cy < 0.75:
            filtered.append(seg)
    
    print(f"Nombre d'objets dans '{region}' : {len(filtered)}")
    for seg in filtered:
        print(f"  - ID {seg['segment_id']} à {seg['centroid']}")

def example_5_extract_object(json_path, image_path, segment_id, output_path="extracted_object.png"):
    """Exemple 5 : Extraire un objet spécifique"""
    print(f"\n=== EXEMPLE 5 : Extraire le segment {segment_id} ===")
    
    # Charger les données
    sam_data = load_sam_json(json_path)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Trouver le segment
    segment = None
    for seg in sam_data['segments']:
        if seg['segment_id'] == segment_id:
            segment = seg
            break
    
    if segment is None:
        print(f"❌ Segment {segment_id} non trouvé")
        return
    
    # Décoder le masque
    if sam_data['format'] == 'rle':
        mask = mask_utils.decode(segment['mask_rle'])
    else:
        mask = np.array(segment['mask_binary'])
    
    # Extraire l'objet (avec fond transparent)
    object_rgba = np.dstack([image, mask * 255])
    
    # Sauvegarder
    plt.figure(figsize=(10, 10))
    plt.imshow(object_rgba)
    plt.axis('off')
    plt.title(f"Segment {segment_id} - {segment['area_ratio']*100:.2f}%")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', transparent=True)
    plt.close()
    
    print(f"✓ Objet extrait sauvegardé : {output_path}")

def example_6_draw_bboxes(json_path, image_path, output_path="with_bboxes.jpg"):
    """Exemple 6 : Dessiner toutes les bounding boxes"""
    print("\n=== EXEMPLE 6 : Dessiner les bounding boxes ===")
    
    # Charger les données
    sam_data = load_sam_json(json_path)
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    
    # Dessiner chaque bbox
    for seg in sam_data['segments']:
        # Convertir bbox normalisée en pixels
        x, y, w, h = seg['bbox']
        x_px = int(x * width)
        y_px = int(y * height)
        w_px = int(w * width)
        h_px = int(h * height)
        
        # Dessiner la box
        color = (0, 255, 0)  # Vert
        cv2.rectangle(image, (x_px, y_px), (x_px + w_px, y_px + h_px), color, 2)
        
        # Ajouter l'ID et la taille
        label = f"#{seg['segment_id']} ({seg['area_ratio']*100:.1f}%)"
        cv2.putText(image, label, (x_px, y_px - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Sauvegarder
    cv2.imwrite(output_path, image)
    print(f"✓ Image avec bboxes sauvegardée : {output_path}")

def example_7_overlay_masks(json_path, image_path, output_path="overlay.png"):
    """Exemple 7 : Superposer tous les masques colorés"""
    print("\n=== EXEMPLE 7 : Superposer les masques ===")
    
    # Charger les données
    sam_data = load_sam_json(json_path)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Créer la figure
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    ax.imshow(image)
    
    # Superposer chaque masque avec une couleur aléatoire
    for seg in sam_data['segments']:
        if sam_data['format'] == 'rle':
            mask = mask_utils.decode(seg['mask_rle'])
        else:
            mask = np.array(seg['mask_binary'])
        
        # Couleur aléatoire
        color = np.random.rand(3)
        
        # Créer un masque coloré
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
        for i in range(3):
            colored_mask[:, :, i] = color[i]
        
        # Superposer avec transparence
        ax.imshow(np.dstack((colored_mask, mask * 0.4)))
    
    ax.axis('off')
    ax.set_title(f"{sam_data['num_segments']} segments", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Overlay sauvegardé : {output_path}")

def example_8_statistics(json_path):
    """Exemple 8 : Statistiques sur les segments"""
    print("\n=== EXEMPLE 8 : Statistiques ===")
    
    sam_data = load_sam_json(json_path)
    segments = sam_data['segments']
    
    areas = [seg['area_ratio'] for seg in segments]
    
    print(f"Nombre total de segments : {len(segments)}")
    print(f"Aire moyenne : {np.mean(areas)*100:.2f}%")
    print(f"Aire médiane : {np.median(areas)*100:.2f}%")
    print(f"Plus petit segment : {np.min(areas)*100:.2f}%")
    print(f"Plus grand segment : {np.max(areas)*100:.2f}%")
    
    # Distribution par taille
    tiny = sum(1 for a in areas if a < 0.01)
    small = sum(1 for a in areas if 0.01 <= a < 0.05)
    medium = sum(1 for a in areas if 0.05 <= a < 0.2)
    large = sum(1 for a in areas if a >= 0.2)
    
    print("\nDistribution par taille :")
    print(f"  Très petits (< 1%) : {tiny}")
    print(f"  Petits (1-5%) : {small}")
    print(f"  Moyens (5-20%) : {medium}")
    print(f"  Grands (> 20%) : {large}")

def example_9_combine_segments(json_path, image_path, segment_ids, output_path="combined.png"):
    """Exemple 9 : Combiner plusieurs segments"""
    print(f"\n=== EXEMPLE 9 : Combiner les segments {segment_ids} ===")
    
    # Charger les données
    sam_data = load_sam_json(json_path)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Créer un masque combiné
    combined_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    
    for seg_id in segment_ids:
        # Trouver le segment
        segment = next((s for s in sam_data['segments'] if s['segment_id'] == seg_id), None)
        if segment is None:
            print(f"⚠️  Segment {seg_id} non trouvé")
            continue
        
        # Décoder et combiner
        if sam_data['format'] == 'rle':
            mask = mask_utils.decode(segment['mask_rle'])
        else:
            mask = np.array(segment['mask_binary'])
        
        combined_mask = np.logical_or(combined_mask, mask).astype(np.uint8)
    
    # Visualiser
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    axes[0].imshow(image)
    axes[0].set_title("Image originale")
    axes[0].axis('off')
    
    axes[1].imshow(image)
    axes[1].imshow(combined_mask, alpha=0.5, cmap='jet')
    axes[1].set_title(f"Segments combinés : {segment_ids}")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Segments combinés sauvegardés : {output_path}")

def example_10_export_to_custom_format(json_path, output_path="custom_export.json"):
    """Exemple 10 : Exporter vers un format personnalisé"""
    print("\n=== EXEMPLE 10 : Export personnalisé ===")
    
    sam_data = load_sam_json(json_path)
    
    # Créer un format personnalisé (exemple : pour annotation)
    custom_export = {
        "image_info": {
            "width": sam_data['image_size'][0],
            "height": sam_data['image_size'][1]
        },
        "objects": []
    }
    
    for seg in sam_data['segments']:
        # Extraire seulement les infos nécessaires
        obj = {
            "id": seg['segment_id'],
            "class": "unknown",  # À remplir manuellement
            "bbox": {
                "x": seg['bbox'][0],
                "y": seg['bbox'][1],
                "width": seg['bbox'][2],
                "height": seg['bbox'][3]
            },
            "center": {
                "x": seg['centroid'][0],
                "y": seg['centroid'][1]
            },
            "area_percentage": round(seg['area_ratio'] * 100, 2)
        }
        custom_export['objects'].append(obj)
    
    # Sauvegarder
    with open(output_path, 'w') as f:
        json.dump(custom_export, f, indent=2)
    
    print(f"✓ Export personnalisé sauvegardé : {output_path}")
    print(f"  Format simplifié avec {len(custom_export['objects'])} objets")

def main():
    import sys
    
    print("=" * 60)
    print("  EXEMPLES D'UTILISATION DES JSON SAM")
    print("=" * 60)
    
    if len(sys.argv) < 3:
        print("\n❌ Usage : python examples_json.py <json_path> <image_path>")
        print("\nExemple :")
        print("  python examples_json.py photo_sam_output.json photo.jpg")
        return
    
    json_path = sys.argv[1]
    image_path = sys.argv[2]
    
    # Vérifier que les fichiers existent
    import os
    if not os.path.exists(json_path):
        print(f"❌ Fichier JSON non trouvé : {json_path}")
        return
    if not os.path.exists(image_path):
        print(f"❌ Image non trouvée : {image_path}")
        return
    
    # Exécuter tous les exemples
    try:
        example_1_count_objects(json_path)
        example_2_largest_objects(json_path, top_n=3)
        example_3_filter_by_size(json_path, min_area=0.05, max_area=0.3)
        example_4_filter_by_position(json_path, region="center")
        example_5_extract_object(json_path, image_path, segment_id=0)
        example_6_draw_bboxes(json_path, image_path)
        example_7_overlay_masks(json_path, image_path)
        example_8_statistics(json_path)
        example_9_combine_segments(json_path, image_path, segment_ids=[0, 1, 2])
        example_10_export_to_custom_format(json_path)
        
        print("\n" + "=" * 60)
        print("✅ Tous les exemples ont été exécutés !")
        print("=" * 60)
        print("\nFichiers générés :")
        print("  - extracted_object.png")
        print("  - with_bboxes.jpg")
        print("  - overlay.png")
        print("  - combined.png")
        print("  - custom_export.json")
        
    except Exception as e:
        print(f"\n❌ Erreur : {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
