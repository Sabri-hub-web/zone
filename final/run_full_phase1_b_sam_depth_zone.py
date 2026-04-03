"""
run_full_phase1_b_sam_depth_zone.py

Pipeline Phase 1 (option B) :
  1) SAM_for_paysagea : preprocess -> SAM -> *_preprocessed.json + *_sam_output.json + image préprocessée
  2) Depth-Anything : run_depth_paysagea.py sur l'image préprocessée
  3) fuse_sam_depth.py : VisionOutput.json (segments + depth)
  4) export_vision_v0.py : main.json (VisionOutput v0)
  5) zone-selection : UI (polygon/brush) export user_zone.json + mask/overlay
  6) merge_user_zone_with_vision.py : integration/final_scene_input.json

Objectif : tout s'aligne sur la MÊME image préprocessée (référence) pour éviter les décalages.

Usage :
  python run_full_phase1_b_sam_depth_zone.py --image Inputs/ton_image.jpg
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


def run_cmd(cmd: list[str], cwd: Optional[Path] = None, env: Optional[dict[str, str]] = None) -> None:
    cwd_str = str(cwd) if cwd else os.getcwd()
    print(f"\n[RUN] cwd={cwd_str}\n  " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    # Windows: évite les UnicodeEncodeError (console cp1252) quand les scripts printent des emojis.
    merged_env.setdefault("PYTHONUTF8", "1")
    merged_env.setdefault("PYTHONIOENCODING", "utf-8")

    subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=merged_env, check=True)


def latest_file_by_mtime(pattern: str, directory: Path) -> Path:
    files = list(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Aucun fichier trouvé avec pattern='{pattern}' dans {directory}")
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    project_root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Phase1-B SAM+Depth+Zone selection+VisionOutput v0")
    parser.add_argument("--image", required=True, type=str, help="Image brute (jpeg/png) dans Paysagea/Inputs ou chemin absolu.")
    parser.add_argument("--sam-checkpoint", default="", type=str, help="Optionnel : chemin vers checkpoint SAM (pth).")
    parser.add_argument("--sam-target-output-dir", default="pipeline_results", type=str, help="Dossier SAM (dans SAM_for_paysagea).")
    parser.add_argument("--depth-outdir", default="Outputs", type=str, help="Dossier de sortie Depth-Anything.")
    parser.add_argument("--zone-display-scale", default=3.0, type=float, help="Facteur d'upscale pour l'UI zone-selection.")
    parser.add_argument(
        "--skip-zone-selection",
        action="store_true",
        help="Si activé, n'ouvre pas l'UI zone-selection et utilise --user-zone-json.",
    )
    parser.add_argument(
        "--user-zone-json",
        default="",
        type=str,
        help="Chemin (absolu ou relatif) vers user_zone.json à utiliser quand --skip-zone-selection est activé.",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = (project_root / "Inputs" / image_path).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image introuvable : {image_path}")

    sam_dir = project_root / "SAM_for_paysagea"
    depth_dir = project_root / "Depth-Anything"
    zone_app_dir = project_root / "zone-selection" / "app"
    zone_outputs_dir = project_root / "zone-selection" / "outputs"
    integration_dir = project_root / "integration"

    sam_results_dir = sam_dir / args.sam_target_output_dir
    inputs_dir = project_root / "Inputs"

    # =========================
    # 1) SAM automation
    # =========================
    sam_start = time.time()

    cmd = [
        sys.executable,
        str(sam_dir / "auto_pipeline" / "pipeline_auto.py"),
        str(image_path),
        "--output",
        str(args.sam_target_output_dir),
    ]
    if args.sam_checkpoint:
        sam_checkpoint_path = Path(args.sam_checkpoint).expanduser()
        if not sam_checkpoint_path.is_absolute():
            # Résout depuis la racine Paysagea (cwd supposée au lancement de l'orchestrateur)
            sam_checkpoint_path = (project_root / sam_checkpoint_path).resolve()
        sam_checkpoint_path = sam_checkpoint_path.resolve()
        if not sam_checkpoint_path.exists():
            raise FileNotFoundError(f"sam-checkpoint introuvable : {sam_checkpoint_path}")
        cmd += ["--checkpoint", str(sam_checkpoint_path)]

    run_cmd(cmd, cwd=sam_dir)

    # On récupère les outputs SAM les plus récents
    preprocessed_json = latest_file_by_mtime("*_preprocessed.json", sam_results_dir)
    preprocess_meta = load_json(preprocessed_json)

    preprocessed_filename = preprocess_meta["preprocessed_filename"]  # ex: menphis_depay-400x225.jpg
    preprocessed_img_path = (sam_results_dir / preprocessed_filename).resolve()
    if not preprocessed_img_path.exists():
        raise FileNotFoundError(f"Préprocessed image introuvable : {preprocessed_img_path}")

    # Pour éviter des décalages côté frontend (qui s'attend souvent à retrouver les images dans Inputs/),
    # on copie l'image préprocessée dans Inputs/ et on utilise cette copie comme référence UI.
    inputs_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_img_in_inputs = (inputs_dir / preprocessed_filename).resolve()
    if preprocessed_img_in_inputs.exists():
        # overwrite sûr : on veut la version la plus récente
        preprocessed_img_in_inputs.unlink(missing_ok=True)
    preprocessed_img_in_inputs.write_bytes(preprocessed_img_path.read_bytes())

    base_name = preprocessed_img_path.stem
    sam_json = (sam_results_dir / f"{base_name}_sam_output.json").resolve()
    if not sam_json.exists():
        raise FileNotFoundError(f"SAM output introuvable : {sam_json}")

    print("\n[OK] SAM outputs détectés :")
    print(f"  preprocessed_img_path: {preprocessed_img_path}")
    print(f"  preprocessed_img_in_inputs: {preprocessed_img_in_inputs}")
    print(f"  preprocessed_json    : {preprocessed_json}")
    print(f"  sam_json             : {sam_json}")

    # =========================
    # 2) Depth Anything
    # =========================
    depth_outdir = depth_dir / args.depth_outdir
    depth_outdir.mkdir(parents=True, exist_ok=True)

    run_cmd(
        [
            sys.executable,
            str(depth_dir / "run_depth_paysagea.py"),
            "--img",
            str(preprocessed_img_path),
            "--meta",
            str(preprocessed_json),
            "--outdir",
            str(depth_outdir),
        ],
        cwd=depth_dir,
    )

    # Nom attendu par run_depth_paysagea.py
    # base = img_path.stem.replace("_preprocessed","").replace("_01","")
    depth_base = base_name.replace("_preprocessed", "").replace("_01", "")
    depth_npy = (depth_outdir / f"{depth_base}_depth.npy").resolve()
    depth_json = (depth_outdir / f"{depth_base}_depth.json").resolve()
    if not depth_npy.exists() or not depth_json.exists():
        # fallback : prendre les plus récents
        depth_npy = latest_file_by_mtime(f"*_depth.npy", depth_outdir)
        depth_json_candidates = list(depth_outdir.glob(f"{depth_npy.stem.replace('_depth','')}_depth.json"))
        depth_json = latest_file_by_mtime("*_depth.json", depth_outdir)

    print("\n[OK] Depth outputs détectés :")
    print(f"  depth_npy : {depth_npy}")
    print(f"  depth_json: {depth_json}")

    # =========================
    # 3) Fuse SAM + Depth => VisionOutput.json
    # =========================
    vision_out = depth_dir / "VisionOutput.json"
    masks_dir = depth_outdir / "masks"

    run_cmd(
        [
            sys.executable,
            str(depth_dir / "fuse_sam_depth.py"),
            "--sam-json",
            str(sam_json),
            "--depth-npy",
            str(depth_npy),
            "--depth-json",
            str(depth_json),
            "--preprocess-json",
            str(preprocessed_json),
            "--out-json",
            str(vision_out),
            "--out-masks-dir",
            str(masks_dir),
        ],
        cwd=depth_dir,
    )

    if not vision_out.exists():
        raise FileNotFoundError(f"VisionOutput.json introuvable : {vision_out}")

    # =========================
    # 4) export VisionOutput v0 => main.json
    # =========================
    main_json_path = project_root / "main.json"
    run_cmd(
        [
            sys.executable,
            str(depth_dir / "export_vision_v0.py"),
            "--input",
            str(vision_out),
            "--output",
            str(main_json_path),
        ],
        cwd=depth_dir,
    )
    if not main_json_path.exists():
        raise FileNotFoundError(f"main.json introuvable : {main_json_path}")

    # =========================
    # 5) Zone-selection UI (interactive)
    # =========================
    zone_outputs_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_zone_selection:
        user_zone_json = None
        if args.user_zone_json:
            candidate = Path(args.user_zone_json).expanduser()
            if not candidate.is_absolute():
                candidate = (project_root / candidate).resolve()
            candidate = candidate.resolve()
            if not candidate.exists():
                raise FileNotFoundError(f"--user-zone-json introuvable : {candidate}")
            user_zone_json = candidate
        else:
            # Par défaut : utiliser le dernier user_zone.json produit.
            candidate = (zone_outputs_dir / "user_zone.json").resolve()
            if not candidate.exists():
                raise FileNotFoundError(
                    "user_zone.json introuvable pour skip zone-selection. "
                    "Soit désactive skip, soit passe --user-zone-json."
                )
            user_zone_json = candidate
        print(f"\n[OK] Skip zone-selection => user_zone_json: {user_zone_json}")
    else:
        run_cmd(
            [
                sys.executable,
                str(zone_app_dir / "main.py"),
                "--ref-image",
                str(preprocessed_img_in_inputs),
                "--output-dir",
                str(zone_outputs_dir),
                "--display-scale",
                str(args.zone_display_scale),
            ],
            cwd=zone_app_dir,
        )

        user_zone_json = zone_outputs_dir / "user_zone.json"
        if not user_zone_json.exists():
            raise FileNotFoundError(f"user_zone.json introuvable : {user_zone_json}")

    # =========================
    # 6) Merge => final_scene_input.json
    # =========================
    final_out = integration_dir / "final_scene_input.json"
    run_cmd(
        [
            sys.executable,
            str(integration_dir / "merge_user_zone_with_vision.py"),
            "--vision",
            str(main_json_path),
            "--user-zone",
            str(user_zone_json),
            "--out",
            str(final_out),
        ],
        cwd=project_root,
    )

    print("\n[OK] Pipeline Phase1-B terminee.")
    print(f"  final_scene_input.json: {final_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

