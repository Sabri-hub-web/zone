"""
generate_garden_cli.py
======================
Script CLI appele par le backend Node.js pour generer l'image du jardin
via BFL FLUX Fill PRO, en exploitant les donnees SAM + Depth du pipeline.

Flux:
  1. Charge pipeline_result.json (SAM + Depth)
  2. Charge user_zone.json (zone plantable selectionnee)
  3. Identifie les segments SAM qui intersectent la zone
  4. Trie par profondeur (fond -> avant plane, back -> front)
  5. Construit le masque BFL = union des segments dans la zone
  6. Appel BFL inpaint avec le masque + prompt
  7. Post-processing (feathered composite)
  8. Genere les masques individuels par segment (mode edition)
  9. Retourne JSON avec output_url + plant_masks[]

Usage:
    python generate_garden_cli.py \\
        --work-dir     <chemin/work/> \\
        --pipeline-json <chemin/xxxxx_pipeline_result.json> \\
        --prompt       "texte optionnel" \\
        --plant-density medium|low|high \\
        --max-plants   5 \\
        --seed         42

Variables d'environnement:
    BFL_API_KEY     cle API Black Forest Labs
    MOCK_BFL=true   simuler sans appel API
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

# Ajouter le dossier garden_ia_3 au path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))


# ---------------------------------------------------------------------------
# Utilitaires RLE (Run-Length Encoding COCO)
# ---------------------------------------------------------------------------

def rle_to_mask(rle_counts: str, height: int, width: int) -> np.ndarray:
    """
    Decode un masque compresse en RLE COCO (format string) en tableau numpy (height, width).
    Retourne un masque binaire uint8 (0/255).
    Utilise pycocotools si disponible, sinon implémentation manuelle.
    """
    try:
        from pycocotools import mask as coco_mask
        rle = {"counts": rle_counts.encode(), "size": [height, width]}
        decoded = coco_mask.decode(rle)  # returns uint8 array of 0/1
        return (decoded * 255).astype(np.uint8)
    except ImportError:
        pass

    # Fallback manuel : décodage COCO RLE string (delta-encoded, 5 bits/byte)
    # Chaque nombre est delta-encodé par rapport au précédent (sauf le premier).
    # Bit 5 (32) = continuation, bits 0-4 = données, bit 4 dernier octet = signe.
    counts = []
    p = 0
    s = rle_counts
    while p < len(s):
        x = 0
        k = 0
        more = True
        while more:
            c = ord(s[p]) - 48
            p += 1
            more = bool(c & 32)
            x |= (c & 31) << (5 * k)
            k += 1
            if not more and (c & 16):
                x |= -(1 << (5 * k))
        if k > 1 and counts:
            x += counts[-1]
        counts.append(x)

    mask = np.zeros(height * width, dtype=np.uint8)
    pos = 0
    fill = 0
    for cnt in counts:
        if fill and pos + cnt <= len(mask):
            mask[pos:pos + cnt] = 255
        pos += cnt
        fill = 1 - fill

    return mask.reshape((height, width), order='F')


def decode_segment_mask(seg: dict, img_w: int, img_h: int) -> np.ndarray:
    """
    Decode le mask_rle d'un segment en masque numpy (img_h, img_w) uint8.
    Gere aussi les masques bitmap (liste de 0/1).
    """
    mask_info = seg.get("mask_rle") or seg.get("segmentation") or {}

    if isinstance(mask_info, dict):
        size = mask_info.get("size", [img_h, img_w])
        counts = mask_info.get("counts", "")
        rle_h, rle_w = size[0], size[1]
        if isinstance(counts, str) and counts:
            try:
                mask = rle_to_mask(counts, rle_h, rle_w)
                # Redimensionner si necessaire
                if mask.shape != (img_h, img_w):
                    pil = Image.fromarray(mask).resize((img_w, img_h), Image.NEAREST)
                    mask = np.array(pil)
                return mask
            except Exception as e:
                print(f"  [RLE] Erreur decode segment {seg.get('segment_id', '?')}: {e}")
        elif isinstance(counts, list):
            # Format bitmap COCO
            pos = 0
            arr = np.zeros(rle_h * rle_w, dtype=np.uint8)
            fill = 0
            for cnt in counts:
                if fill and pos + cnt <= len(arr):
                    arr[pos:pos + cnt] = 255
                pos += cnt
                fill = 1 - fill
            mask = arr.reshape((rle_h, rle_w), order='F')
            if mask.shape != (img_h, img_w):
                pil = Image.fromarray(mask).resize((img_w, img_h), Image.NEAREST)
                mask = np.array(pil)
            return mask

    # Fallback: masque vide
    return np.zeros((img_h, img_w), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Construction du masque BFL depuis la zone utilisateur + segments SAM
# ---------------------------------------------------------------------------

def build_user_zone_mask(zone_data: dict, img_w: int, img_h: int) -> np.ndarray:
    """
    Construit le masque polygonal depuis user_zone.json.
    Retourne un masque uint8 (0/255).
    """
    normalized = zone_data.get("normalized_points") or zone_data.get("points_normalized")
    absolute = zone_data.get("points")

    mask_np = np.zeros((img_h, img_w), dtype=np.uint8)

    if normalized and len(normalized) >= 3:
        points_px = [
            (int(round(p["x"] * img_w)), int(round(p["y"] * img_h)))
            for p in normalized
        ]
        print(f"  [ZONE] {len(points_px)} points normalises -> pixels ({img_w}x{img_h})")
    elif absolute and len(absolute) >= 3:
        points_px = [(int(p["x"]), int(p["y"])) for p in absolute]
        print(f"  [ZONE] {len(points_px)} points absolus (px)")
    else:
        print("  [ZONE] Pas de points valides — fallback: 50% inferior de l'image")
        sky_cut = int(img_h * 0.40)
        mask_np[sky_cut:, :] = 255
        return mask_np

    pil = Image.fromarray(mask_np, mode="L")
    draw = ImageDraw.Draw(pil)
    draw.polygon(points_px, fill=255)
    mask_np = np.array(pil)
    white_pct = 100.0 * np.sum(mask_np >= 128) / mask_np.size
    print(f"  [ZONE] Masque polygonal: {white_pct:.1f}% de l'image")
    return mask_np


def select_segments_in_zone(
    segments: list[dict],
    zone_mask: np.ndarray,
    img_w: int,
    img_h: int,
    max_plants: int = 8,
    min_intersection_ratio: float = 0.05,
) -> list[dict]:
    """
    Selectionne les segments SAM qui intersectent la zone plantable.
    Les trie par profondeur mean_depth croissante (fond -> avant).

    Retourne une liste de dicts:
        { segment_id, mean_depth, depth_band, bbox, centroid, mask_pixels, intersection_ratio }
    """
    zone_bin = zone_mask >= 128
    zone_area = np.sum(zone_bin)
    if zone_area == 0:
        print("  [SEG] Zone vide — utilisation de tous les segments")
        zone_bin = np.ones((img_h, img_w), dtype=bool)

    selected = []
    for seg in segments:
        seg_mask = decode_segment_mask(seg, img_w, img_h)
        seg_bin = seg_mask >= 128

        intersection = np.sum(seg_bin & zone_bin)
        if intersection == 0:
            continue

        seg_area = np.sum(seg_bin)
        intersection_ratio = intersection / seg_area if seg_area > 0 else 0.0

        zone_coverage = intersection / zone_area if zone_area > 0 else 0.0
        if intersection_ratio < min_intersection_ratio and zone_coverage < 0.20:
            # Segment trop peu dans la zone ET couvre moins de 20% de la zone
            continue

        selected.append({
            "segment_id": seg["segment_id"],
            "mean_depth": seg.get("mean_depth", 0.5),
            "depth_band": seg.get("depth_band", "mid"),
            "area_ratio": seg.get("area_ratio", 0.0),
            "bbox": seg.get("bbox", [0, 0, 1, 1]),
            "centroid": seg.get("centroid", [0.5, 0.5]),
            "intersection_pixels": int(intersection),
            "intersection_ratio": round(intersection_ratio, 3),
            "mask_pixels": seg_mask,  # gardé en mémoire pour le masquage
        })

    # Trier par mean_depth croissante: fond (depth faible) -> avant (depth ~1.0)
    # near_is_one=True donc: depth proche de 0 = loin, proche de 1 = pres
    # On veut peindre du fond vers l'avant: sort ascending depth
    selected.sort(key=lambda s: s["mean_depth"])

    print(f"  [SEG] {len(selected)} segments intersectent la zone")
    for s in selected[:max_plants]:
        print(f"     seg {s['segment_id']:2d} depth={s['mean_depth']:.3f} "
              f"band={s['depth_band']} area={s['area_ratio']:.3f} "
              f"inter={s['intersection_ratio']:.2f}")

    # Limiter au nombre max de plantes
    return selected[:max_plants]


def build_bfl_mask_from_segments(
    segments_selected: list[dict],
    zone_mask: np.ndarray,
    img_w: int,
    img_h: int,
    out_path: Path,
) -> tuple[np.ndarray, Path]:
    """
    Construit le masque BFL = union des masques de segments dans la zone.
    Si pas de segments: utilise directement le masque de zone.

    BFL convention: blanc=modifier, noir=conserver.
    """
    if segments_selected:
        combined = np.zeros((img_h, img_w), dtype=np.uint8)
        for seg in segments_selected:
            seg_mask = seg["mask_pixels"]
            zone_bin = (zone_mask >= 128).astype(np.uint8)
            # Intersection segment ∩ zone
            intersection = np.where((seg_mask >= 128) & (zone_bin == 1), 255, 0).astype(np.uint8)
            combined = np.maximum(combined, intersection)
        method = "sam_segments"
    else:
        combined = np.where(zone_mask >= 128, 255, 0).astype(np.uint8)
        method = "zone_only"

    # Binariser strictement (0/255)
    combined = np.where(combined >= 128, 255, 0).astype(np.uint8)
    white_pct = 100.0 * np.sum(combined == 255) / combined.size
    print(f"  [MASK] Masque BFL ({method}): {white_pct:.1f}% blanc")

    mask_pil = Image.fromarray(combined, mode="L")
    mask_pil.save(out_path)
    return combined, out_path


def save_individual_masks(
    segments_selected: list[dict],
    zone_mask: np.ndarray,
    img_w: int,
    img_h: int,
    masks_dir: Path,
) -> list[dict]:
    """
    Sauvegarde un masque PNG par segment SAM (pour le mode edition frontend).
    Retourne la liste des infos de masques.
    """
    masks_info = []
    masks_dir.mkdir(parents=True, exist_ok=True)
    zone_bin = zone_mask >= 128

    for i, seg in enumerate(segments_selected):
        seg_mask = seg["mask_pixels"]
        # Masque du segment intersecte avec la zone
        plant_mask = np.where((seg_mask >= 128) & zone_bin, 255, 0).astype(np.uint8)
        fname = f"plant_mask_{i:02d}_seg{seg['segment_id']}.png"
        fpath = masks_dir / fname
        Image.fromarray(plant_mask, mode="L").save(fpath)

        masks_info.append({
            "plant_index": i,
            "segment_id": seg["segment_id"],
            "depth_band": seg["depth_band"],
            "mean_depth": round(seg["mean_depth"], 4),
            "area_ratio": seg["area_ratio"],
            "centroid": seg["centroid"],
            "bbox": seg["bbox"],
            "mask_file": fname,
            "mask_url": f"/work/masks/{fname}",
        })

    print(f"  [MASKS] {len(masks_info)} masques individuels saves dans {masks_dir.name}/")
    return masks_info


# ---------------------------------------------------------------------------
# Construction du prompt
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

def build_prompt(plant_density: str, user_description: str, segments: list[dict], work_dir: Path) -> str:
    """Construit le prompt optimise pour BFL FLUX Fill PRO avec contexte RAG si possible."""
    
    # ── 1. Charger contexte manifeste ──
    latest_proj = Path(work_dir).parent / "shared" / "latest_project.json"
    manifest = {}
    if latest_proj.exists():
        try:
            with open(latest_proj, encoding="utf-8") as f:
                manifest = json.load(f)
        except:
            pass

    env = manifest.get("environmental_context", {})
    location = env.get("location", {}).get("short_label", "France")
    style = manifest.get("user_intent", {}).get("applied_tags", [])
    if not style:
        style = [manifest.get("user_intent", {}).get("description", "naturel")]

    # ── 2. Charger imports prompt_builder ──
    try:
        from image_generation.prompt_builder import (
            ADDITIVE_BASE, ADDITIVE_VISIBLE, ADDITIVE_NEGATIVE,
            MASK_CONSTRAINT, PLACEMENT_REALISTIC,
        )
    except ImportError:
        ADDITIVE_BASE = ("Use the input photo as the base. Keep camera angle, perspective, "
                         "lighting, shadows, and all existing elements exactly the same. "
                         "Preserve the original photo completely.")
        ADDITIVE_VISIBLE = ("Add clearly noticeable plants ONLY in the masked areas. "
                            "The added plants must be clearly visible and recognizable. "
                            "Keep image sharp, no blur, no haze.")
        ADDITIVE_NEGATIVE = ("DO NOT add any new massive objects like walls, paths or benches. "
                             "Only add the requested plants. DO NOT add people or animals.")
        MASK_CONSTRAINT = "ONLY paint inside the masked area. Outside the mask must remain pixel-identical to the original."
        PLACEMENT_REALISTIC = "Plants must be grounded in soil, with realistic scale and consistent shadows."

    # Relaxer ADDITIVE_NEGATIVE pour BFL
    # Si on est trop strict ("no mulch, no gravel"), BFL ignore les plantes car il ne peut pas poser de terre dessous.
    relaxed_negative = "DO NOT add any human-made objects like paths, benches, ponds, lights, fences, or walls. DO NOT add people or animals."

    density_map = {
        "low": "a few carefully placed ornamental plants",
        "medium": "several clearly visible flowering plants and shrubs",
        "high": "lush, dense planting with many colorful plants and garden textures",
    }
    density_desc = density_map.get(plant_density, "several clearly visible plants")

    # ── 3. Charger les plantes depuis rag_output.json (produit par le RAG) ──
    plant_block = ""   # bloc mis EN TÊTE de prompt si RAG dispo
    rag_active = False
    try:
        rag_path = Path(work_dir) / "rag_output.json"
        if rag_path.exists():
            from image_generation.utils_rag import load_rag
            from image_generation.prompt_builder import _get_visual
            _, plants = load_rag(rag_path)
            PLANT_TYPES = {"arbuste", "fleur", "vivace", "graminee", "arbre", "rosier", "haie", "couvre_sol"}
            valid_plants = [
                p for p in plants
                if p.get("name") and p.get("type", "").lower() in PLANT_TYPES
            ][:6]
            if valid_plants:
                # Nom + description visuelle courte pour chaque plante
                entries = []
                for p in valid_plants:
                    name = p["name"]
                    visual = _get_visual(p)
                    color = (p.get("color") or "").replace("|", "/")
                    color_hint = f", {color} color" if color else ""
                    entries.append(f"{name} ({visual[:80]}{color_hint})")
                plant_block = "ADD THESE SPECIFIC PLANTS clearly visible in the masked area: " + "; ".join(entries) + "."
                rag_active = True
                print(f"[RAG] {len(valid_plants)} plantes chargées pour le prompt")
    except Exception as e:
        print(f"[RAG] Erreur chargement rag_output.json: {e}")

    # Contexte de profondeur
    depth_hint = ""
    if segments:
        bands = [s["depth_band"] for s in segments]
        if "mid" in bands or "back" in bands:
            depth_hint = " Maintain natural depth perspective and scale variation between foreground and background plants."

    desc_hint = ""
    if user_description and user_description.strip():
        desc_hint = f" Ambiance: {user_description.strip()[:200]}."

    # Contexte géographique et style depuis latest_project.json
    context_hint = ""
    if location and location != "France":
        context_hint += f" Garden located in {location}."
    if style:
        style_str = style[0] if isinstance(style, list) and style else str(style)
        if style_str and len(style_str) > 3:
            context_hint += f" Garden style: {style_str[:80]}."

    if rag_active:
        # Prompt structuré : plantes EN TÊTE, puis contraintes de préservation
        prompt = (
            f"{plant_block} "
            f"{ADDITIVE_BASE} "
            f"{MASK_CONSTRAINT} "
            f"{PLACEMENT_REALISTIC} "
            f"{depth_hint}{desc_hint}{context_hint} "
            f"{relaxed_negative}"
        )
    else:
        # Fallback sans RAG : prompt générique
        prompt = (
            f"{ADDITIVE_BASE} "
            f"{ADDITIVE_VISIBLE} "
            f"{density_desc}.{depth_hint}{desc_hint}{context_hint} "
            f"{MASK_CONSTRAINT} "
            f"{PLACEMENT_REALISTIC} "
            f"{relaxed_negative}"
        )
    return " ".join(prompt.split())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Genere jardin via BFL + pipeline SAM/Depth")
    parser.add_argument("--work-dir", required=True, help="Dossier work/ (contient pipeline_result.json, user_zone.json, etc.)")
    parser.add_argument("--pipeline-json", default="", help="Chemin complet vers xxxxx_pipeline_result.json (auto-detect si vide)")
    parser.add_argument("--prompt", default="", help="Description utilisateur")
    parser.add_argument("--plant-density", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--max-plants", type=int, default=6, help="Nombre max de segments a utiliser")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    masks_dir = work_dir / "masks"

    # ── 1. Trouver le pipeline_result.json ────────────────────────────────────
    if args.pipeline_json and Path(args.pipeline_json).exists():
        pipeline_path = Path(args.pipeline_json)
    else:
        # Auto-detect: prendre le plus recent pipeline_result.json dans work/
        candidates = sorted(
            work_dir.glob("*_pipeline_result.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            result = {"status": "error", "error": "Aucun fichier *_pipeline_result.json trouve dans work/."}
            print(json.dumps(result, ensure_ascii=False))
            sys.exit(1)
        pipeline_path = candidates[0]
        print(f"[PIPELINE] Auto-detect: {pipeline_path.name}")

    # ── 2. Verifications des inputs ───────────────────────────────────────────
    zone_path = work_dir / "user_zone.json"
    if not zone_path.exists():
        result = {"status": "error", "error": "user_zone.json introuvable dans work/"}
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(1)

    # ── 3. Charger les donnees ────────────────────────────────────────────────
    print(f"\n[GEN] Pipeline JSON : {pipeline_path.name}")
    print(f"[GEN] Zone JSON     : {zone_path.name}")
    print(f"[GEN] Density       : {args.plant_density}  Max plants: {args.max_plants}  Seed: {args.seed}")

    with open(pipeline_path, encoding="utf-8") as f:
        pipeline_data = json.load(f)
    with open(zone_path, encoding="utf-8") as f:
        zone_data = json.load(f)

    # Extraire les metadonnees
    vision = pipeline_data.get("vision", {})
    image_size = vision.get("image_size") or pipeline_data.get("depth", {}).get("image_size", [768, 1024])
    img_w, img_h = image_size[0], image_size[1]

    # Trouver l'image preprocessee
    files_meta = pipeline_data.get("files", {})
    preprocess_img_name = files_meta.get("preprocessed_image", "")
    preprocess_path = work_dir / preprocess_img_name if preprocess_img_name else None

    # Fallback: chercher une image jpg correspondante
    if not preprocess_path or not preprocess_path.exists():
        img_candidates = sorted(work_dir.glob("*_preprocessed.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
        if img_candidates:
            preprocess_path = img_candidates[0]
            print(f"  [IMG] Auto-detect image: {preprocess_path.name}")
        else:
            result = {"status": "error", "error": "Image preprocessee introuvable dans work/"}
            print(json.dumps(result, ensure_ascii=False))
            sys.exit(1)

    print(f"[GEN] Image source  : {preprocess_path.name} ({img_w}x{img_h})")

    # ── 4. Segments SAM ───────────────────────────────────────────────────────
    segments = vision.get("segments", [])
    sam_meta = vision.get("sam_meta", {})
    print(f"\n[SAM] {len(segments)} segments charges (total SAM: {sam_meta.get('segments_count', '?')})")

    # ── 5. Masque de zone utilisateur ─────────────────────────────────────────
    print("\n[ZONE] Construction du masque de zone plantable...")
    zone_mask = build_user_zone_mask(zone_data, img_w, img_h)
    zone_mask_path = work_dir / "user_zone_mask_bfl.png"
    Image.fromarray(zone_mask, mode="L").save(zone_mask_path)

    # ── 6. Selectionner segments dans la zone ─────────────────────────────────
    print("\n[SEG] Selection des segments SAM dans la zone plantable...")
    segments_selected = select_segments_in_zone(
        segments, zone_mask, img_w, img_h,
        max_plants=args.max_plants,
        min_intersection_ratio=0.05,
    )

    # ── 7. Masque BFL = union segments ∩ zone ────────────────────────────────
    print("\n[MASK] Construction du masque BFL...")
    bin_path = masks_dir / "plantable_mask_bin.png"
    masks_dir.mkdir(exist_ok=True)
    bfl_mask_arr, bin_path = build_bfl_mask_from_segments(
        segments_selected, zone_mask, img_w, img_h, bin_path
    )
    white_pct_bin = 100.0 * np.sum(bfl_mask_arr == 255) / bfl_mask_arr.size

    # Sanity check: si masque > 65% → reduire par erosion
    if white_pct_bin > 65.0:
        print(f"  [MASK] Masque trop large ({white_pct_bin:.1f}%) -> erosion")
        from PIL import ImageFilter
        pil_m = Image.fromarray(bfl_mask_arr, mode="L")
        for _ in range(10):
            pil_m = pil_m.filter(ImageFilter.MinFilter(3))
        bfl_mask_arr = np.array(pil_m)
        bfl_mask_arr = np.where(bfl_mask_arr >= 128, 255, 0).astype(np.uint8)
        Image.fromarray(bfl_mask_arr, mode="L").save(bin_path)
        white_pct_bin = 100.0 * np.sum(bfl_mask_arr == 255) / bfl_mask_arr.size
        print(f"  [MASK] Apres erosion: {white_pct_bin:.1f}%")

    # ── 8. Sauvegarder les masques individuels ────────────────────────────────
    print("\n[MASKS] Sauvegarde des masques individuels (mode edition)...")
    masks_info = save_individual_masks(segments_selected, zone_mask, img_w, img_h, masks_dir)

    # ── 9. Prompt ─────────────────────────────────────────────────────────────
    prompt = build_prompt(args.plant_density, args.prompt, segments_selected, work_dir)
    print(f"\n[PROMPT] {prompt[:150]}...")
    (work_dir / "last_prompt.txt").write_text(prompt, encoding="utf-8")

    # ── 10. Appel BFL ─────────────────────────────────────────────────────────
    bfl_key = os.environ.get("BFL_API_KEY", "").strip()
    use_mock = os.environ.get("MOCK_BFL", "").lower() == "true" or not bfl_key
    final_path = work_dir / "final_garden.png"

    if use_mock:
        print("\n[GEN] MODE MOCK — copie image source (aucun appel BFL)")
        import shutil
        shutil.copy(preprocess_path, final_path)
        gen_status = "mock"
    else:
        print(f"\n[GEN] BFL_API_KEY: presente ({len(bfl_key)} chars)")
        print("[GEN] Lancement inpainting BFL FLUX Fill PRO...")
        try:
            from image_generation.bfl_provider import inpaint
            from image_generation.config import BFL_GUIDANCE, BFL_STEPS, BFL_STRENGTH

            # Strength dynamique selon taille masque
            strength = BFL_STRENGTH
            if white_pct_bin > 55:
                strength = min(strength, 0.65)
            elif white_pct_bin > 40:
                strength = min(strength, 0.70)
            print(f"  BFL params: steps={BFL_STEPS} guidance={BFL_GUIDANCE} strength={strength} seed={args.seed}")

            inpaint(
                image_path=preprocess_path,
                mask_path=str(bin_path),
                prompt=prompt,
                out_path=final_path,
                seed=args.seed,
                steps=BFL_STEPS,
                guidance=BFL_GUIDANCE,
                strength=strength,
            )
            print(f"  BFL termine -> {final_path.name}")

            # Post-processing: feathered composite (original hors masque + genere dans masque)
            if final_path.exists():
                try:
                    from scipy.ndimage import gaussian_filter
                    orig = Image.open(preprocess_path).convert("RGB")
                    gen = Image.open(final_path).convert("RGB")
                    if gen.size != orig.size:
                        gen = gen.resize(orig.size, Image.LANCZOS)
                    orig_arr = np.array(orig, dtype=np.float32)
                    gen_arr = np.array(gen, dtype=np.float32)
                    mask_float = bfl_mask_arr.astype(np.float32) / 255.0
                    alpha = gaussian_filter(mask_float, sigma=4.0)
                    alpha = np.clip(alpha, 0.0, 1.0)[..., np.newaxis]
                    combined = (orig_arr * (1.0 - alpha) + gen_arr * alpha).astype(np.uint8)
                    Image.fromarray(combined).save(final_path)
                    print("  Post-processing: fusion feathered appliquee")
                except ImportError:
                    print("  scipy absent - feathering ignore (BFL direct)")
                except Exception as e:
                    print(f"  Post-processing echoue: {e}")

            gen_status = "success"
        except RuntimeError as e:
            result = {"status": "error", "error": str(e)}
            print(f"[GEN] ERREUR BFL: {e}", file=sys.stderr)
            print(json.dumps(result, ensure_ascii=False))
            sys.exit(1)

    # ── 11. Construire le resultat JSON ───────────────────────────────────────
    result = {
        "status": gen_status,
        "output_filename": "final_garden.png",
        "output_path": str(final_path),
        "image_size": [img_w, img_h],
        "pipeline_json_used": pipeline_path.name,
        "segments_used": len(segments_selected),
        "mask_white_pct": round(white_pct_bin, 1),
        "prompt_preview": prompt[:250],
        "seed": args.seed,
        "plant_masks": [
            {k: v for k, v in m.items() if k != "mask_pixels"}
            for m in masks_info
        ],
    }

    # Sauvegarder le JSON dans work/
    result_path = work_dir / "generate_garden_result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Print JSON sur stdout pour Node.js
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
