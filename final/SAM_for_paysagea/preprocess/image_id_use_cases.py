#!/usr/bin/env python3
"""
image_id_use_cases.py
=====================
Démonstration des cas d'usage de l'image_id pour la fusion et le tracking.
"""

import json
from pathlib import Path
from PIL import Image

from preprocess_image import preprocess_image, load_metadata


def scenario_1_stability():
    """
    SCÉNARIO 1 : Stabilité de l'image_id
    =====================================
    Même image source → même image_id (utile pour déduplication)
    """
    print("=" * 70)
    print("SCÉNARIO 1 : STABILITÉ DE L'IMAGE_ID")
    print("=" * 70)
    
    # Créer une image de test
    img = Image.new('RGB', (1920, 1080), color='blue')
    img.save('test_photo.jpg')
    
    print("\n📸 Prétraitement de la même image 3 fois...")
    
    # Prétraiter 3 fois
    metadata1 = preprocess_image('test_photo.jpg', 'out1.jpg', max_side=1024)
    metadata2 = preprocess_image('test_photo.jpg', 'out2.jpg', max_side=1024)
    metadata3 = preprocess_image('test_photo.jpg', 'out3.jpg', max_side=1024)
    
    id1 = metadata1["image_id"]
    id2 = metadata2["image_id"]
    id3 = metadata3["image_id"]
    
    print(f"\n✅ Résultats :")
    print(f"   Image 1 : {id1}")
    print(f"   Image 2 : {id2}")
    print(f"   Image 3 : {id3}")
    
    if id1 == id2 == id3:
        print(f"\n🎉 Tous les image_id sont identiques !")
        print(f"   → Utile pour détecter les duplicatas")
        print(f"   → Utile pour la mise en cache")
    
    print()


def scenario_2_detection():
    """
    SCÉNARIO 2 : Détection de modifications
    ========================================
    Image modifiée → image_id différent
    """
    print("=" * 70)
    print("SCÉNARIO 2 : DÉTECTION DE MODIFICATIONS")
    print("=" * 70)
    
    # Version 1 : Image bleue
    img_v1 = Image.new('RGB', (1920, 1080), color='blue')
    img_v1.save('photo_v1.jpg')
    
    metadata_v1 = preprocess_image('photo_v1.jpg', 'out_v1.jpg', max_side=1024)
    id_v1 = metadata_v1["image_id"]
    
    print(f"\n📸 Version 1 (bleue) : {id_v1}")
    
    # Version 2 : Image rouge (modifiée)
    img_v2 = Image.new('RGB', (1920, 1080), color='red')
    img_v2.save('photo_v2.jpg')
    
    metadata_v2 = preprocess_image('photo_v2.jpg', 'out_v2.jpg', max_side=1024)
    id_v2 = metadata_v2["image_id"]
    
    print(f"📸 Version 2 (rouge)  : {id_v2}")
    
    if id_v1 != id_v2:
        print(f"\n✅ Les image_id sont différents !")
        print(f"   → Détection automatique des modifications")
        print(f"   → Invalidation de cache possible")
    
    print()


def scenario_3_pipeline_fusion():
    """
    SCÉNARIO 3 : Fusion de résultats de plusieurs pipelines
    ========================================================
    Utiliser l'image_id pour garantir qu'on fusionne les bons résultats
    """
    print("=" * 70)
    print("SCÉNARIO 3 : FUSION DE RÉSULTATS DE PIPELINES")
    print("=" * 70)
    
    # Créer une image
    img = Image.new('RGB', (1920, 1080), color='green')
    img.save('sample.jpg')
    
    # Prétraiter
    metadata = preprocess_image('sample.jpg', 'sample_preprocessed.jpg', max_side=1024)
    image_id = metadata["image_id"]
    
    print(f"\n🔑 Image ID : {image_id}")
    
    # Simuler 3 pipelines différents travaillant sur la même image
    
    # Pipeline 1 : SAM2 détection
    sam2_results = {
        "image_id": image_id,
        "pipeline": "SAM2",
        "timestamp": "2026-02-03T10:00:00Z",
        "boxes": [
            {"id": 1, "bbox": [100, 50, 300, 400], "label": "person"},
            {"id": 2, "bbox": [400, 200, 700, 450], "label": "car"}
        ]
    }
    
    # Pipeline 2 : Classification
    classifier_results = {
        "image_id": image_id,
        "pipeline": "ResNet",
        "timestamp": "2026-02-03T10:01:30Z",
        "scene": "outdoor",
        "confidence": 0.92
    }
    
    # Pipeline 3 : OCR
    ocr_results = {
        "image_id": image_id,
        "pipeline": "Tesseract",
        "timestamp": "2026-02-03T10:02:15Z",
        "text_regions": [
            {"text": "STOP", "bbox": [50, 100, 150, 150]}
        ]
    }
    
    print(f"\n📊 Résultats collectés :")
    print(f"   1. SAM2 : {len(sam2_results['boxes'])} objets détectés")
    print(f"   2. Classification : scène '{classifier_results['scene']}'")
    print(f"   3. OCR : {len(ocr_results['text_regions'])} région(s) de texte")
    
    # Fusion sécurisée grâce à l'image_id
    def safe_merge(*results):
        """Fusionne les résultats seulement s'ils ont le même image_id."""
        image_ids = [r["image_id"] for r in results]
        
        if len(set(image_ids)) != 1:
            raise ValueError(f"❌ Image IDs différents : {image_ids}")
        
        merged = {
            "image_id": image_ids[0],
            "pipelines": {}
        }
        
        for result in results:
            pipeline_name = result["pipeline"]
            merged["pipelines"][pipeline_name] = {
                k: v for k, v in result.items() 
                if k not in ["image_id", "pipeline"]
            }
        
        return merged
    
    print(f"\n🔄 Fusion des résultats...")
    merged = safe_merge(sam2_results, classifier_results, ocr_results)
    
    print(f"\n✅ Résultats fusionnés pour image_id : {merged['image_id']}")
    print(json.dumps(merged, indent=2))
    
    print(f"\n💡 Avantages :")
    print(f"   ✓ Garantit que tous les résultats concernent la même image")
    print(f"   ✓ Détecte automatiquement les erreurs de mapping")
    print(f"   ✓ Permet la fusion asynchrone de pipelines distribués")
    
    print()


def scenario_4_batch_tracking():
    """
    SCÉNARIO 4 : Tracking de batch d'images
    ========================================
    Gérer plusieurs images avec leurs métadonnées
    """
    print("=" * 70)
    print("SCÉNARIO 4 : TRACKING DE BATCH D'IMAGES")
    print("=" * 70)
    
    # Créer un batch de 3 images
    images = [
        ('img1.jpg', 'red'),
        ('img2.jpg', 'green'),
        ('img3.jpg', 'blue'),
    ]
    
    print("\n📸 Création et prétraitement de 3 images...")
    
    batch_metadata = []
    
    for filename, color in images:
        # Créer l'image
        img = Image.new('RGB', (1920, 1080), color=color)
        img.save(filename)
        
        # Prétraiter
        output = f"{Path(filename).stem}_preprocessed.jpg"
        metadata = preprocess_image(filename, output, max_side=1024)
        
        batch_metadata.append(metadata)
        
        print(f"   ✓ {filename} → {metadata['image_id']}")
    
    # Créer un registre
    registry = {
        "batch_id": "batch_2026_02_03_001",
        "timestamp": "2026-02-03T10:00:00Z",
        "total_images": len(batch_metadata),
        "images": {
            meta["image_id"]: {
                "source_filename": meta["source_filename"],
                "preprocessed_filename": meta["preprocessed_filename"],
                "dimensions": meta["preprocess"]["resized_size"]
            }
            for meta in batch_metadata
        }
    }
    
    print(f"\n📋 Registre du batch :")
    print(json.dumps(registry, indent=2))
    
    print(f"\n💡 Cas d'usage :")
    print(f"   ✓ Tracking de batches de traitement")
    print(f"   ✓ Audit trail complet")
    print(f"   ✓ Réconciliation des résultats")
    
    print()


def scenario_5_distributed_system():
    """
    SCÉNARIO 5 : Système distribué
    ===============================
    Utiliser l'image_id pour coordonner des traitements distribués
    """
    print("=" * 70)
    print("SCÉNARIO 5 : SYSTÈME DISTRIBUÉ")
    print("=" * 70)
    
    # Créer une image
    img = Image.new('RGB', (1920, 1080), color='orange')
    img.save('distributed_test.jpg')
    
    # Serveur A : Prétraitement
    print("\n🖥️  SERVEUR A : Prétraitement")
    metadata = preprocess_image(
        'distributed_test.jpg',
        'distributed_test_preprocessed.jpg',
        max_side=1024
    )
    
    image_id = metadata["image_id"]
    print(f"   Image ID généré : {image_id}")
    print(f"   → Envoi à la queue de traitement")
    
    # Message dans une queue (Redis, RabbitMQ, etc.)
    task_message = {
        "task_id": "task_123",
        "image_id": image_id,
        "preprocessed_url": "s3://bucket/distributed_test_preprocessed.jpg",
        "operations": ["detect", "classify", "ocr"]
    }
    
    print(f"\n📨 Message dans la queue :")
    print(json.dumps(task_message, indent=2))
    
    # Serveur B : Détection
    print(f"\n🖥️  SERVEUR B : Détection SAM2")
    detection_result = {
        "task_id": "task_123",
        "image_id": image_id,  # ⭐ Référence stable
        "operation": "detect",
        "status": "completed",
        "results": {"boxes": [...]}
    }
    print(f"   Traitement de image_id : {image_id}")
    print(f"   ✓ Détection terminée")
    
    # Serveur C : Classification
    print(f"\n🖥️  SERVEUR C : Classification")
    classification_result = {
        "task_id": "task_123",
        "image_id": image_id,  # ⭐ Référence stable
        "operation": "classify",
        "status": "completed",
        "results": {"class": "outdoor", "confidence": 0.95}
    }
    print(f"   Traitement de image_id : {image_id}")
    print(f"   ✓ Classification terminée")
    
    # Serveur D : Agrégation
    print(f"\n🖥️  SERVEUR D : Agrégation des résultats")
    
    def aggregate_results(task_id, expected_image_id):
        """Agrège les résultats d'une tâche distribuée."""
        # Récupérer tous les résultats pour cette tâche
        results = [detection_result, classification_result]
        
        # Vérifier que tous concernent la même image
        image_ids = [r["image_id"] for r in results]
        if not all(iid == expected_image_id for iid in image_ids):
            raise ValueError(f"❌ Incohérence : image_ids = {image_ids}")
        
        print(f"   ✅ Tous les résultats concernent bien {expected_image_id}")
        
        return {
            "task_id": task_id,
            "image_id": expected_image_id,
            "status": "completed",
            "results": {r["operation"]: r["results"] for r in results}
        }
    
    final_result = aggregate_results("task_123", image_id)
    
    print(f"\n📊 Résultat final :")
    print(json.dumps(final_result, indent=2))
    
    print(f"\n💡 Avantages dans un système distribué :")
    print(f"   ✓ Référence stable entre serveurs")
    print(f"   ✓ Pas besoin de base de données centralisée")
    print(f"   ✓ Détection d'erreurs de routing")
    print(f"   ✓ Traçabilité complète")
    
    print()


if __name__ == "__main__":
    print("\n" + "🎯" * 35)
    print("DÉMONSTRATION : CAS D'USAGE DE L'IMAGE_ID")
    print("🎯" * 35)
    print()
    
    scenario_1_stability()
    scenario_2_detection()
    scenario_3_pipeline_fusion()
    scenario_4_batch_tracking()
    scenario_5_distributed_system()
    
    print("=" * 70)
    print("RÉSUMÉ DES AVANTAGES DE L'IMAGE_ID")
    print("=" * 70)
    print("""
1. 🔑 IDENTIFIANT UNIQUE ET STABLE
   - Basé sur le contenu de l'image preprocessée
   - Même image → même ID (déduplication, cache)
   - Image différente → ID différent (détection de modification)

2. 🔄 FUSION DE PIPELINES
   - Garantit qu'on fusionne les résultats de la bonne image
   - Prévient les erreurs de mapping
   - Permet la composition asynchrone de résultats

3. 📊 TRACKING ET AUDIT
   - Suivi de batches d'images
   - Traçabilité complète du workflow
   - Réconciliation facilitée

4. 🌐 SYSTÈMES DISTRIBUÉS
   - Référence stable entre serveurs
   - Pas de dépendance à une DB centrale
   - Détection d'erreurs de routing

5. ⚡ PERFORMANCE
   - Mise en cache basée sur le contenu
   - Évite le retraitement d'images identiques
   - Optimisation des pipelines
    """)
    
    print("🎉" * 35)
