#!/usr/bin/env python3
"""
test_preprocess_image.py
========================
Tests unitaires pour preprocess_image.py
"""

import json
import tempfile
from pathlib import Path

import pytest
from PIL import Image

from preprocess_image import (
    preprocess_image,
    save_metadata,
    load_metadata,
    convert_coordinates_to_original,
    convert_coordinates_to_resized
)


@pytest.fixture
def temp_dir():
    """Crée un répertoire temporaire pour les tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_image_large(temp_dir):
    """Crée une image de test 1920x1080."""
    img_path = temp_dir / "large.jpg"
    img = Image.new('RGB', (1920, 1080), color='red')
    img.save(img_path)
    return img_path


@pytest.fixture
def sample_image_small(temp_dir):
    """Crée une image de test 800x600."""
    img_path = temp_dir / "small.jpg"
    img = Image.new('RGB', (800, 600), color='blue')
    img.save(img_path)
    return img_path


@pytest.fixture
def sample_image_rgba(temp_dir):
    """Crée une image RGBA de test."""
    img_path = temp_dir / "rgba.png"
    img = Image.new('RGBA', (1024, 768), color=(255, 0, 0, 128))
    img.save(img_path)
    return img_path


class TestPreprocessImage:
    """Tests pour la fonction preprocess_image."""
    
    def test_resize_large_image(self, temp_dir, sample_image_large):
        """Test : redimensionnement d'une grande image."""
        output_path = temp_dir / "output.jpg"
        
        metadata = preprocess_image(
            str(sample_image_large),
            str(output_path),
            max_side=1024
        )
        
        # Vérifier les dimensions
        assert metadata["preprocess"]["original_size"] == [1920, 1080]
        assert metadata["preprocess"]["resized_size"] == [1024, 576]
        
        # Vérifier le scale_factor
        expected_scale = 1024 / 1920
        assert abs(metadata["preprocess"]["scale_factor"] - expected_scale) < 0.0001
        
        # Vérifier que l'image existe
        assert output_path.exists()
        
        # Vérifier les dimensions de l'image sauvegardée
        img = Image.open(output_path)
        assert img.size == (1024, 576)
    
    def test_no_upscale_small_image(self, temp_dir, sample_image_small):
        """Test : pas d'upscale pour une petite image."""
        output_path = temp_dir / "output.jpg"
        
        metadata = preprocess_image(
            str(sample_image_small),
            str(output_path),
            max_side=1024
        )
        
        # L'image est déjà < 1024, pas de redimensionnement
        assert metadata["preprocess"]["original_size"] == [800, 600]
        assert metadata["preprocess"]["resized_size"] == [800, 600]
        assert metadata["preprocess"]["scale_factor"] == 1.0
        
        # Vérifier l'image
        img = Image.open(output_path)
        assert img.size == (800, 600)
    
    def test_rgba_to_rgb_conversion(self, temp_dir, sample_image_rgba):
        """Test : conversion RGBA → RGB."""
        output_path = temp_dir / "output.jpg"
        
        metadata = preprocess_image(
            str(sample_image_rgba),
            str(output_path),
            max_side=1024
        )
        
        # Vérifier que l'image est RGB
        img = Image.open(output_path)
        assert img.mode == 'RGB'
    
    def test_keep_ratio(self, temp_dir, sample_image_large):
        """Test : le ratio d'aspect est préservé."""
        output_path = temp_dir / "output.jpg"
        
        metadata = preprocess_image(
            str(sample_image_large),
            str(output_path),
            max_side=1024
        )
        
        orig_w, orig_h = metadata["preprocess"]["original_size"]
        new_w, new_h = metadata["preprocess"]["resized_size"]
        
        orig_ratio = orig_w / orig_h
        new_ratio = new_w / new_h
        
        # Le ratio doit être préservé (tolérance 0.01%)
        assert abs(orig_ratio - new_ratio) < 0.0001
    
    def test_metadata_structure(self, temp_dir, sample_image_large):
        """Test : structure des métadonnées."""
        output_path = temp_dir / "output.jpg"
        
        metadata = preprocess_image(
            str(sample_image_large),
            str(output_path),
            max_side=1024
        )
        
        # Vérifier la structure au niveau racine
        assert "image_id" in metadata
        assert "source_filename" in metadata
        assert "preprocessed_filename" in metadata
        assert "preprocess" in metadata
        
        # Vérifier les champs de tracking
        assert metadata["image_id"].startswith("sha256:")
        assert metadata["source_filename"] == "large.jpg"
        assert metadata["preprocessed_filename"] == "output.jpg"
        
        # Vérifier la structure preprocess
        preprocess = metadata["preprocess"]
        
        assert "original_size" in preprocess
        assert "resized_size" in preprocess
        assert "scale_factor" in preprocess
        assert "max_side" in preprocess
        assert "keep_ratio" in preprocess
        assert "orientation" in preprocess
        
        # Vérifier la structure orientation
        orientation = preprocess["orientation"]
        assert "exif_present" in orientation
        assert "exif_orientation" in orientation
        assert "applied_rotation_deg" in orientation
        
        # Vérifier les types
        assert isinstance(metadata["image_id"], str)
        assert isinstance(metadata["source_filename"], str)
        assert isinstance(metadata["preprocessed_filename"], str)
        assert isinstance(preprocess["original_size"], list)
        assert isinstance(preprocess["resized_size"], list)
        assert isinstance(preprocess["scale_factor"], (int, float))
        assert isinstance(preprocess["max_side"], int)
        assert isinstance(preprocess["keep_ratio"], bool)
        assert isinstance(orientation["exif_present"], bool)
        assert isinstance(orientation["applied_rotation_deg"], int)
        # exif_orientation peut être None
        if orientation["exif_orientation"] is not None:
            assert isinstance(orientation["exif_orientation"], int)


class TestMetadataIO:
    """Tests pour save/load metadata."""
    
    def test_save_and_load_metadata(self, temp_dir):
        """Test : sauvegarde et chargement des métadonnées."""
        metadata = {
            "image_id": "sha256:a3f2e8b1c9d4f5e6",
            "source_filename": "photo.jpg",
            "preprocessed_filename": "photo_preprocessed.jpg",
            "preprocess": {
                "original_size": [1920, 1080],
                "resized_size": [1024, 576],
                "scale_factor": 0.5333,
                "max_side": 1024,
                "keep_ratio": True,
                "orientation": {
                    "exif_present": True,
                    "exif_orientation": 6,
                    "applied_rotation_deg": 270
                }
            }
        }
        
        metadata_path = temp_dir / "metadata.json"
        
        # Sauvegarder
        save_metadata(metadata, str(metadata_path))
        assert metadata_path.exists()
        
        # Charger
        loaded = load_metadata(str(metadata_path))
        assert loaded == metadata
    
    def test_metadata_json_format(self, temp_dir):
        """Test : format JSON valide."""
        metadata = {
            "image_id": "sha256:b7c3d2a1e4f8b9d0",
            "source_filename": "test.jpg",
            "preprocessed_filename": "test_preprocessed.jpg",
            "preprocess": {
                "original_size": [1920, 1080],
                "resized_size": [1024, 576],
                "scale_factor": 0.5333,
                "max_side": 1024,
                "keep_ratio": True,
                "orientation": {
                    "exif_present": False,
                    "exif_orientation": None,
                    "applied_rotation_deg": 0
                }
            }
        }
        
        metadata_path = temp_dir / "metadata.json"
        save_metadata(metadata, str(metadata_path))
        
        # Vérifier que c'est du JSON valide
        with open(metadata_path) as f:
            loaded = json.load(f)
        
        assert loaded == metadata


class TestCoordinateConversion:
    """Tests pour la conversion de coordonnées."""
    
    @pytest.fixture
    def metadata(self):
        """Métadonnées de test."""
        return {
            "image_id": "sha256:test123456789abc",
            "source_filename": "test.jpg",
            "preprocessed_filename": "test_preprocessed.jpg",
            "preprocess": {
                "original_size": [1920, 1080],
                "resized_size": [1024, 576],
                "scale_factor": 0.5333,
                "max_side": 1024,
                "keep_ratio": True,
                "orientation": {
                    "exif_present": False,
                    "exif_orientation": None,
                    "applied_rotation_deg": 0
                }
            }
        }
    
    def test_convert_to_original(self, metadata):
        """Test : conversion resized → original."""
        # Point au centre de l'image redimensionnée
        x_resized, y_resized = 512, 288
        
        x_orig, y_orig = convert_coordinates_to_original(
            x_resized, y_resized, metadata
        )
        
        # Doit être proche du centre de l'image originale
        assert abs(x_orig - 960) < 1.0
        assert abs(y_orig - 540) < 1.0
    
    def test_convert_to_resized(self, metadata):
        """Test : conversion original → resized."""
        # Point au centre de l'image originale
        x_orig, y_orig = 960, 540
        
        x_resized, y_resized = convert_coordinates_to_resized(
            x_orig, y_orig, metadata
        )
        
        # Doit être proche du centre de l'image redimensionnée
        assert abs(x_resized - 512) < 1.0
        assert abs(y_resized - 288) < 1.0
    
    def test_conversion_roundtrip(self, metadata):
        """Test : conversion aller-retour."""
        # Coordonnées originales
        x_orig, y_orig = 1000, 600
        
        # Convertir vers resized
        x_resized, y_resized = convert_coordinates_to_resized(
            x_orig, y_orig, metadata
        )
        
        # Convertir de retour vers original
        x_back, y_back = convert_coordinates_to_original(
            x_resized, y_resized, metadata
        )
        
        # Doit être identique (tolérance pour les flottants)
        assert abs(x_back - x_orig) < 0.1
        assert abs(y_back - y_orig) < 0.1
    
    def test_corner_coordinates(self, metadata):
        """Test : conversion des coins de l'image."""
        # Coin supérieur gauche de l'image redimensionnée
        x_orig, y_orig = convert_coordinates_to_original(0, 0, metadata)
        assert x_orig == 0
        assert y_orig == 0
        
        # Coin inférieur droit de l'image redimensionnée
        w, h = metadata["preprocess"]["resized_size"]
        x_orig, y_orig = convert_coordinates_to_original(w, h, metadata)
        
        orig_w, orig_h = metadata["preprocess"]["original_size"]
        assert abs(x_orig - orig_w) < 1.0
        assert abs(y_orig - orig_h) < 1.0


class TestImageID:
    """Tests pour l'identifiant d'image."""
    
    def test_image_id_format(self, temp_dir, sample_image_large):
        """Test : format de l'image_id."""
        output_path = temp_dir / "output.jpg"
        
        metadata = preprocess_image(
            str(sample_image_large),
            str(output_path),
            max_side=1024
        )
        
        image_id = metadata["image_id"]
        
        # Doit commencer par "sha256:"
        assert image_id.startswith("sha256:")
        
        # Doit avoir 16 caractères après "sha256:"
        hash_part = image_id.split(":", 1)[1]
        assert len(hash_part) == 16
        
        # Doit être en hexadécimal
        assert all(c in "0123456789abcdef" for c in hash_part)
    
    def test_image_id_stability(self, temp_dir, sample_image_large):
        """Test : même image → même image_id."""
        output1 = temp_dir / "output1.jpg"
        output2 = temp_dir / "output2.jpg"
        
        # Prétraiter deux fois la même image
        metadata1 = preprocess_image(
            str(sample_image_large),
            str(output1),
            max_side=1024
        )
        
        metadata2 = preprocess_image(
            str(sample_image_large),
            str(output2),
            max_side=1024
        )
        
        # Les image_id doivent être identiques
        assert metadata1["image_id"] == metadata2["image_id"]
    
    def test_image_id_uniqueness(self, temp_dir):
        """Test : images différentes → image_id différents."""
        # Créer deux images différentes
        img1_path = temp_dir / "img1.jpg"
        img2_path = temp_dir / "img2.jpg"
        
        img1 = Image.new('RGB', (1920, 1080), color='red')
        img2 = Image.new('RGB', (1920, 1080), color='blue')
        
        img1.save(img1_path)
        img2.save(img2_path)
        
        output1 = temp_dir / "output1.jpg"
        output2 = temp_dir / "output2.jpg"
        
        # Prétraiter
        metadata1 = preprocess_image(str(img1_path), str(output1), max_side=1024)
        metadata2 = preprocess_image(str(img2_path), str(output2), max_side=1024)
        
        # Les image_id doivent être différents
        assert metadata1["image_id"] != metadata2["image_id"]
    
    def test_image_id_changes_with_different_max_side(self, temp_dir, sample_image_large):
        """Test : différent max_side → différent image_id."""
        output1 = temp_dir / "output1.jpg"
        output2 = temp_dir / "output2.jpg"
        
        # Prétraiter avec différents max_side
        metadata1 = preprocess_image(
            str(sample_image_large),
            str(output1),
            max_side=1024
        )
        
        metadata2 = preprocess_image(
            str(sample_image_large),
            str(output2),
            max_side=512
        )
        
        # Les images preprocessées sont différentes → image_id différents
        assert metadata1["image_id"] != metadata2["image_id"]


class TestFileTracking:
    """Tests pour le tracking des noms de fichiers."""
    
    def test_source_filename_tracking(self, temp_dir):
        """Test : tracking du nom de fichier source."""
        img_path = temp_dir / "my_photo_2024.jpg"
        img = Image.new('RGB', (1000, 800), color='green')
        img.save(img_path)
        
        output_path = temp_dir / "output.jpg"
        metadata = preprocess_image(
            str(img_path),
            str(output_path),
            max_side=1024
        )
        
        assert metadata["source_filename"] == "my_photo_2024.jpg"
    
    def test_preprocessed_filename_tracking(self, temp_dir, sample_image_large):
        """Test : tracking du nom de fichier preprocessé."""
        output_path = temp_dir / "IMG_5177_preprocessed.jpg"
        
        metadata = preprocess_image(
            str(sample_image_large),
            str(output_path),
            max_side=1024
        )
        
        assert metadata["preprocessed_filename"] == "IMG_5177_preprocessed.jpg"


class TestEdgeCases:
    """Tests pour les cas limites."""
    
    def test_image_without_exif(self, temp_dir):
        """Test : image sans métadonnées EXIF."""
        img_path = temp_dir / "no_exif.jpg"
        img = Image.new('RGB', (1000, 800), color='red')
        img.save(img_path)
        
        output_path = temp_dir / "output.jpg"
        metadata = preprocess_image(
            str(img_path),
            str(output_path),
            max_side=1024
        )
        
        # Vérifier que EXIF est absent
        orientation = metadata["preprocess"]["orientation"]
        assert orientation["exif_present"] == False
        assert orientation["exif_orientation"] is None
        assert orientation["applied_rotation_deg"] == 0
    
    def test_square_image(self, temp_dir):
        """Test : image carrée."""
        img_path = temp_dir / "square.jpg"
        img = Image.new('RGB', (1000, 1000), color='green')
        img.save(img_path)
        
        output_path = temp_dir / "output.jpg"
        metadata = preprocess_image(
            str(img_path),
            str(output_path),
            max_side=800
        )
        
        # L'image doit être redimensionnée à 800x800
        assert metadata["preprocess"]["resized_size"] == [800, 800]
        assert metadata["preprocess"]["scale_factor"] == 0.8
    
    def test_portrait_orientation(self, temp_dir):
        """Test : image en portrait (hauteur > largeur)."""
        img_path = temp_dir / "portrait.jpg"
        img = Image.new('RGB', (600, 1200), color='yellow')
        img.save(img_path)
        
        output_path = temp_dir / "output.jpg"
        metadata = preprocess_image(
            str(img_path),
            str(output_path),
            max_side=800
        )
        
        # La hauteur est le côté le plus long
        assert metadata["preprocess"]["resized_size"][1] == 800
        # La largeur doit être proportionnelle
        assert metadata["preprocess"]["resized_size"][0] == 400
    
    def test_very_small_image(self, temp_dir):
        """Test : image très petite (pas de redimensionnement)."""
        img_path = temp_dir / "tiny.jpg"
        img = Image.new('RGB', (100, 50), color='purple')
        img.save(img_path)
        
        output_path = temp_dir / "output.jpg"
        metadata = preprocess_image(
            str(img_path),
            str(output_path),
            max_side=1024
        )
        
        # Pas de redimensionnement
        assert metadata["preprocess"]["original_size"] == [100, 50]
        assert metadata["preprocess"]["resized_size"] == [100, 50]
        assert metadata["preprocess"]["scale_factor"] == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
