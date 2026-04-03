from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from pycocotools import mask as mask_utils


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_rle_bytes(rle: Dict[str, Any]) -> Dict[str, Any]:
    """
    pycocotools attend généralement "counts" en bytes.
    Notre JSON stocke counts en str (utf-8).
    """
    counts = rle.get("counts")
    if isinstance(counts, str):
        return {"size": rle["size"], "counts": counts.encode("utf-8")}
    return rle


def _union_zones_rle(zones: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    rles = []
    for z in zones:
        rle = z.get("mask_rle")
        if not rle:
            continue
        rles.append(_ensure_rle_bytes(rle))

    if not rles:
        return None

    merged = mask_utils.merge(rles, intersect=False)
    # merged["counts"] est bytes -> on convertit en str pour le JSON
    counts = merged["counts"].decode("utf-8") if isinstance(merged.get("counts"), (bytes, bytearray)) else merged["counts"]
    return {"size": list(merged["size"]), "counts": counts}


def _guess_vision_path(project_root: Path) -> Optional[Path]:
    """
    Cherche un fichier main.json (VisionOutput v0) dans quelques emplacements connus.
    """
    candidates = [
        project_root / "main.json",
        project_root / "Outputs" / "main.json",
        project_root / "Depth-Anything" / "main.json",
        project_root / "Depth-Anything" / "Outputs" / "main.json",
    ]
    for p in candidates:
        if p.exists():
            return p

    # fallback: recherche récursive
    matches = list(project_root.rglob("main.json"))
    if matches:
        # prendre le plus récent (par mtime)
        matches.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return matches[0]

    return None


def merge_user_zone_with_vision(
    vision: Optional[Dict[str, Any]],
    user_zone: Dict[str, Any],
) -> Dict[str, Any]:
    image_id = user_zone.get("image_id")
    image_filename = user_zone.get("image_filename")
    image_size = user_zone.get("image_size")
    zones = user_zone.get("zones", [])

    union_rle = _union_zones_rle(zones)

    payload: Dict[str, Any] = {
        "version": "final_scene_input_v1",
        "created_at": _now_iso_utc(),
        "image": {
            "image_id": image_id,
            "image_filename": image_filename,
            "image_size": image_size,
        },
        "vision": vision,  # peut être None si pas encore généré
        "user_zone": user_zone,
        "user_zone_union": {
            "mask_rle": union_rle,
            "num_zones": len(zones),
        },
    }

    # Avertissement de cohérence (sans bloquer)
    if vision and isinstance(vision, dict):
        v_img_size = (
            vision.get("image_size")
            or (vision.get("image", {}) or {}).get("image_size")
            or (vision.get("image", {}) or {}).get("size")
        )
        if v_img_size and image_size and list(v_img_size) != list(image_size):
            payload["warnings"] = payload.get("warnings", [])
            payload["warnings"].append(
                {
                    "type": "image_size_mismatch",
                    "vision_image_size": v_img_size,
                    "user_zone_image_size": image_size,
                    "message": "Les tailles image de vision et user_zone ne correspondent pas (alignement à vérifier).",
                }
            )

    return payload


def main():
    project_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Fusionne VisionOutput v0 (main.json) avec user_zone.json")
    parser.add_argument(
        "--vision",
        type=str,
        default="",
        help="Chemin vers main.json (VisionOutput v0). Si vide, tentative d'auto-détection.",
    )
    parser.add_argument(
        "--user-zone",
        type=str,
        default=str(project_root / "zone-selection" / "outputs" / "user_zone.json"),
        help="Chemin vers user_zone.json (sortie de l'outil zone-selection).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(project_root / "integration" / "final_scene_input.json"),
        help="Chemin de sortie pour le JSON fusionné.",
    )

    args = parser.parse_args()

    user_zone_path = Path(args.user_zone).resolve()
    if not user_zone_path.exists():
        raise FileNotFoundError(f"user_zone.json introuvable: {user_zone_path}")

    vision_data: Optional[Dict[str, Any]] = None
    vision_path: Optional[Path] = Path(args.vision).resolve() if args.vision else _guess_vision_path(project_root)
    if vision_path and vision_path.exists():
        vision_data = _read_json(vision_path)
        print(f"Vision chargée: {vision_path}")
    else:
        print("Vision non fournie / non détectée. Le champ 'vision' sera null dans la sortie.")

    user_zone_data = _read_json(user_zone_path)
    print(f"user_zone chargé: {user_zone_path}")

    merged = merge_user_zone_with_vision(vision=vision_data, user_zone=user_zone_data)

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"Sortie écrite: {out_path}")


if __name__ == "__main__":
    main()
