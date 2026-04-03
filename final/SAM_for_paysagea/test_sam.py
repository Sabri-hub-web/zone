#!/usr/bin/env python3
"""
Script simple pour tester SAM vit_b
Utilisation : python test_sam.py <image_path>
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import sys
import os

def check_requirements():
    """Vérifie que tout est installé"""
    try:
        import segment_anything
        import cv2
        import matplotlib
        print("✓ Toutes les dépendances sont installées")
        return True
    except ImportError as e:
        print(f"❌ Erreur : {e}")
        print("\nInstallez les dépendances avec :")
        print("pip install opencv-python matplotlib")
        print("pip install git+https://github.com/facebookresearch/segment-anything.git")
        return False

def download_model():
    """Télécharge le modèle vit_b si nécessaire"""
    checkpoint_path = "sam_vit_b_01ec64.pth"
    
    if os.path.exists(checkpoint_path):
        print(f"✓ Modèle trouvé : {checkpoint_path}")
        return checkpoint_path
    
    print("⏳ Téléchargement du modèle vit_b (375 MB)...")
    import urllib.request
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    
    try:
        urllib.request.urlretrieve(url, checkpoint_path)
        print(f"✓ Modèle téléchargé : {checkpoint_path}")
        return checkpoint_path
    except Exception as e:
        print(f"❌ Erreur lors du téléchargement : {e}")
        return None

def load_image(image_path):
    """Charge et prépare l'image"""
    if not os.path.exists(image_path):
        print(f"❌ Erreur : L'image '{image_path}' n'existe pas")
        return None
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Erreur : Impossible de lire '{image_path}'")
        return None
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"✓ Image chargée : {image.shape[1]}x{image.shape[0]} pixels")
    return image

def segment_automatic(image, checkpoint_path, device="cuda"):
    """Segmentation automatique de tous les objets"""
    print("\n=== SEGMENTATION AUTOMATIQUE ===")
    print("Chargement du modèle...")
    
    # Essayer CUDA, sinon CPU
    try:
        import torch
        if device == "cuda" and not torch.cuda.is_available():
            print("⚠️  CUDA non disponible, utilisation du CPU")
            device = "cpu"
    except:
        device = "cpu"
    
    sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    sam.to(device=device)
    print(f"✓ Modèle chargé sur {device}")
    
    # Créer le générateur
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=0,
        min_mask_region_area=100,
    )
    
    print("Génération des masques...")
    masks = mask_generator.generate(image)
    print(f"✓ {len(masks)} objets détectés")
    
    # Visualiser
    visualize_masks(image, masks)
    
    return masks

def segment_with_click(image, checkpoint_path, device="cuda"):
    """Segmentation interactive - cliquez sur l'objet"""
    print("\n=== SEGMENTATION PAR POINT ===")
    print("Chargement du modèle...")
    
    # Essayer CUDA, sinon CPU
    try:
        import torch
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
    except:
        device = "cpu"
    
    sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(image)
    print(f"✓ Modèle chargé sur {device}")
    
    # Interface de clic
    print("\n📍 Cliquez sur l'objet à segmenter")
    print("   - Clic gauche : point positif (sur l'objet)")
    print("   - Clic droit : point négatif (hors de l'objet)")
    print("   - Touche 'q' : terminer et afficher le résultat")
    
    points = []
    labels = []
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image)
    ax.set_title("Cliquez sur l'image (q pour terminer)")
    ax.axis('off')
    
    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            x, y = int(event.xdata), int(event.ydata)
            
            # Clic gauche = positif, clic droit = négatif
            if event.button == 1:  # Clic gauche
                points.append([x, y])
                labels.append(1)
                color = 'green'
                marker = '*'
                print(f"  ✓ Point positif ajouté : ({x}, {y})")
            elif event.button == 3:  # Clic droit
                points.append([x, y])
                labels.append(0)
                color = 'red'
                marker = 'x'
                print(f"  ✗ Point négatif ajouté : ({x}, {y})")
            
            ax.plot(x, y, color=color, marker=marker, markersize=15, markeredgewidth=3)
            plt.draw()
    
    def onkey(event):
        if event.key == 'q' and len(points) > 0:
            plt.close()
    
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('key_press_event', onkey)
    plt.show()
    
    if len(points) == 0:
        print("❌ Aucun point sélectionné")
        return None
    
    # Prédire
    print(f"\nGénération du masque avec {len(points)} point(s)...")
    input_points = np.array(points)
    input_labels = np.array(labels)
    
    masks, scores, _ = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True,
    )
    
    # Afficher les résultats
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for idx, (mask, score, ax) in enumerate(zip(masks, scores, axes)):
        ax.imshow(image)
        ax.imshow(mask, alpha=0.5, cmap='jet')
        
        for point, label in zip(input_points, input_labels):
            color = 'green' if label == 1 else 'red'
            marker = '*' if label == 1 else 'x'
            ax.plot(point[0], point[1], color=color, marker=marker, 
                   markersize=15, markeredgewidth=3)
        
        ax.set_title(f"Masque {idx+1} - Score: {score:.3f}")
        ax.axis('off')
    
    plt.suptitle("Choisissez le meilleur masque", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('segmentation_result.png', dpi=150, bbox_inches='tight')
    print("✓ Résultat sauvegardé : segmentation_result.png")
    plt.show()
    
    return masks, scores

def visualize_masks(image, masks):
    """Visualise les masques générés automatiquement"""
    if len(masks) == 0:
        print("❌ Aucun masque à afficher")
        return
    
    sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    ax.imshow(image)
    ax.set_autoscale_on(False)
    
    for ann in sorted_masks:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))
    
    ax.axis('off')
    ax.set_title(f"{len(masks)} objets détectés", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('automatic_segmentation.png', dpi=150, bbox_inches='tight')
    print("✓ Résultat sauvegardé : automatic_segmentation.png")
    plt.show()

def main():
    print("=" * 60)
    print("  TEST DE SEGMENT ANYTHING MODEL (SAM) - vit_b")
    print("=" * 60)
    
    # Vérifier les prérequis
    if not check_requirements():
        return
    
    # Vérifier l'argument
    if len(sys.argv) < 2:
        print("\n❌ Usage : python test_sam.py <chemin_image>")
        print("Exemple : python test_sam.py photo.jpg")
        return
    
    image_path = sys.argv[1]
    
    # Télécharger le modèle si nécessaire
    checkpoint_path = download_model()
    if checkpoint_path is None:
        return
    
    # Charger l'image
    image = load_image(image_path)
    if image is None:
        return
    
    # Menu interactif
    print("\n" + "=" * 60)
    print("  CHOISISSEZ LE MODE DE SEGMENTATION")
    print("=" * 60)
    print("1. Segmentation automatique (détecte tous les objets)")
    print("2. Segmentation interactive (cliquez sur l'objet)")
    print("3. Les deux")
    
    choice = input("\nVotre choix (1/2/3) : ").strip()
    
    try:
        # Détecter si CUDA est disponible
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("\n⚠️  Utilisation du CPU (peut être lent)")
    except:
        device = "cpu"
    
    if choice == '1':
        segment_automatic(image, checkpoint_path, device)
    elif choice == '2':
        segment_with_click(image, checkpoint_path, device)
    elif choice == '3':
        segment_automatic(image, checkpoint_path, device)
        segment_with_click(image, checkpoint_path, device)
    else:
        print("❌ Choix invalide")
    
    print("\n✅ Terminé !")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interruption par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur : {e}")
        import traceback
        traceback.print_exc()
