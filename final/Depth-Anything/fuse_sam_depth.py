"""
fuse_sam_depth.py - associe la profondeur (Depth-Anything) à chaque segment SAM.

Entrées (par fichier) :
  - --sam-json      : *_sam_output.json (sortie pipeline_auto)
  - --depth-npy     : *_depth.npy (sortie run_depth_paysagea.py)
  - --depth-json    : *_depth.json
  - --preprocess-json (optionnel) : *_preprocessed.json pour enrichir la sortie

Sorties :
  - VisionOutput.json (ou --out-json)
  - un JSON par masque : Outputs/masks/mask_{segment_id}.json (ou --out-masks-dir)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from pycocotools import mask as mask_utils


def depth_band(x: float) -> str:
    """front=proche, mid=milieu, back=loin (x est normalisé 0..1)."""
    if x >= 0.66:
        return "front"
    if x >= 0.33:
        return "mid"
    return "back"


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _decode_rle(rle: Any, h: int, w: int) -> np.ndarray:
    """
    Décode RLE COCO.
    - rle attendu : dict avec "size" + "counts" (counts peut être str).
    """
    if isinstance(rle, dict) and "size" in rle:
        mask = mask_utils.decode(rle)
    else:
        mask = mask_utils.decode({"size": [h, w], "counts": rle})

    # pycocotools renvoie généralement (h,w) mais on sécurise.
    mask_bool = mask.astype(bool)
    if mask_bool.shape == (w, h):
        mask_bool = mask_bool.transpose()
    return mask_bool


def main() -> int:
    parser = argparse.ArgumentParser(description="Fuse SAM segments + DepthAnything depth")
    parser.add_argument("--sam-json", type=str, required=True, help="Chemin vers *_sam_output.json")
    parser.add_argument("--depth-npy", type=str, required=True, help="Chemin vers *_depth.npy")
    parser.add_argument("--depth-json", type=str, required=True, help="Chemin vers *_depth.json")
    parser.add_argument("--preprocess-json", type=str, default="", help="Chemin vers *_preprocessed.json (optionnel)")
    parser.add_argument("--out-json", type=str, default="VisionOutput.json", help="Chemin sortie VisionOutput.json")
    parser.add_argument(
        "--out-masks-dir",
        type=str,
        default="Outputs/masks",
        help="Dossier où écrire mask_{segment_id}.json",
    )
    args = parser.parse_args()

    sam_json = Path(args.sam_json).resolve()
    depth_npy = Path(args.depth_npy).resolve()
    depth_json = Path(args.depth_json).resolve()
    preprocess_json = Path(args.preprocess_json).resolve() if args.preprocess_json else None

    out_json = Path(args.out_json).resolve()
    out_masks_dir = Path(args.out_masks_dir).resolve()

    if not sam_json.exists():
        raise FileNotFoundError(f"sam-json introuvable : {sam_json}")
    if not depth_npy.exists():
        raise FileNotFoundError(f"depth-npy introuvable : {depth_npy}")
    if not depth_json.exists():
        raise FileNotFoundError(f"depth-json introuvable : {depth_json}")
    if preprocess_json and not preprocess_json.exists():
        print(f"⚠️ preprocess-json introuvable (optionnel) : {preprocess_json}")
        preprocess_json = None

    print("Loading depth map...")
    depth = np.load(depth_npy)

    print("Loading depth metadata...")
    depth_meta = _load_json(depth_json)

    print("Loading SAM output...")
    sam_data = _load_json(sam_json)
    segments = sam_data["sam_output"]["segments"]
    print(f"SAM segments: {len(segments)}")

    preprocess_meta: Dict[str, Any] = {}
    if preprocess_json:
        print("Loading preprocess metadata...")
        preprocess_meta = _load_json(preprocess_json)

    H, W = depth.shape

    # Vérif alignement
    sam_size = sam_data["sam_output"].get("image_size")
    if sam_size and sam_size != [W, H]:
        raise ValueError(f"SAM image_size {sam_size} != depth {[W, H]}")

    near_is_one = bool(depth_meta.get("near_is_one", True))

    print("\nComputing depth for each mask...")
    segments_out = []
    for seg in segments:
        seg_id = seg["segment_id"]
        rle = seg["mask_rle"]

        mask = _decode_rle(rle, h=H, w=W)
        if mask.shape != depth.shape:
            raise ValueError(f"Mask shape {mask.shape} != depth {depth.shape}")

        vals = depth[mask]
        if vals.size == 0:
            mean_depth = None
            depth_std = None
            band = None
            min_depth = None
            max_depth = None
        else:
            mean_depth = float(vals.mean())
            depth_std = float(vals.std())
            band = depth_band(mean_depth)
            min_depth = float(vals.min())
            max_depth = float(vals.max())

        seg_enriched = dict(seg)
        seg_enriched["mean_depth"] = mean_depth
        seg_enriched["depth_std"] = depth_std
        seg_enriched["depth_band"] = band
        seg_enriched["min_depth"] = min_depth
        seg_enriched["max_depth"] = max_depth
        seg_enriched["num_pixels"] = int(mask.sum())
        segments_out.append(seg_enriched)

    vision_output = {
        "version": "vision_segments_v1",
        "image_id": preprocess_meta.get("image_id") or sam_data.get("image_id") or depth_meta.get("image_id"),
        "image_size": [W, H],
        "preprocess": preprocess_meta,
        "depth_meta": {
            "model": depth_meta.get("model", "LiheYoung/depth_anything_vitl14"),
            "near_is_one": near_is_one,
            "depth_range": depth_meta.get("depth_range", [0.0, 1.0]),
            "normalized": depth_meta.get("normalized", True),
            "depth_file": str(depth_npy),
        },
        "sam_meta": {
            "sam_file": str(sam_json),
            "segments_count": len(segments_out),
        },
        "segments": segments_out,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(vision_output, f, indent=2)

    out_masks_dir.mkdir(parents=True, exist_ok=True)
    for seg in segments_out:
        seg_id = seg["segment_id"]
        mask_data = {
            "segment_id": seg_id,
            "mean_depth": seg.get("mean_depth"),
            "depth_std": seg.get("depth_std"),
            "depth_band": seg.get("depth_band"),
            "min_depth": seg.get("min_depth"),
            "max_depth": seg.get("max_depth"),
            "num_pixels": seg.get("num_pixels"),
            "area_ratio": seg.get("area_ratio"),
            "centroid": seg.get("centroid"),
            "bbox": seg.get("bbox"),
        }
        mask_file = out_masks_dir / f"mask_{seg_id}.json"
        with mask_file.open("w", encoding="utf-8") as f:
            json.dump(mask_data, f, indent=2)

    print(f"\n✅ Saved {out_json} with {len(segments_out)} segments.")
    print(f"✅ Saved {len(segments_out)} fichiers individuels dans {out_masks_dir}")

    if segments_out:
        example = segments_out[0]
        print("Exemple segment 0:", {k: example.get(k) for k in ["segment_id", "mean_depth", "depth_std", "depth_band"]})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
