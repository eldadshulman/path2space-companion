#!/usr/bin/env python
"""
Numerical-parity check: companion vs Cell_revisions/prediction_pipline/1main_tcga.py.

What this does
--------------
1. Picks a test slide. Default = first TCGA_BRCA row in super_metadata.pkl.
2. Runs the reference 1main_tcga.py on it (CTransPath + grid mode), capturing
   its pred.pkl output. The reference script writes to
       /vf/users/Ruppin_AI/st/Cell_revisions/{project}/prediction/...
   which no longer exists (the Ruppin_AI/st tree was retired in favor of
   Ruppin_ST/). This script copies the reference into a sandbox, rewrites
   every `Ruppin_AI/st` reference to its new `Ruppin_ST` equivalent, and
   redirects `project_dir` to a writable temp location, before running.
3. Runs the companion (Path2SpacePipeline.predict_grid) on the same slide
   with the same TILE_SIZE.
4. Aligns both outputs on (slide_name, x, y) and compares the gene columns
   with `np.allclose(..., atol=1e-5)`.

Exit code 0 on parity, 1 on mismatch.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

REF_SCRIPT = Path("/vf/users/Ruppin_ST/Cell_revisions/prediction_pipline/1main_tcga.py")
SUPER_META = Path("/vf/users/Ruppin_ST/Cell_revisions/super_metadata.pkl")


def pick_test_slide() -> tuple[str, int]:
    """Return (project, i_slide) for the first TCGA_BRCA row of super_metadata."""
    meta = pd.read_pickle(SUPER_META).reset_index(drop=True)
    if "cohort" in meta.columns:
        sub = meta[meta["cohort"] == "TCGA_BRCA"]
        if not len(sub):
            raise RuntimeError("super_metadata has no TCGA_BRCA rows")
        i_slide = int(sub.index[0])
    else:
        i_slide = 0
    project = str(meta.iloc[i_slide].get("cohort", "clinical_cohorts"))
    slide_name = meta.iloc[i_slide]["slide_name"]
    tile_size = int(meta.iloc[i_slide]["tile_size_px_100_micron"])
    slide_path = meta.iloc[i_slide]["path2image"]
    return project, i_slide, slide_name, tile_size, slide_path


def patch_and_run_reference(project: str, i_slide: int, sandbox: Path) -> Path:
    """
    Copy 1main_tcga.py into `sandbox`, redirect its output dir to `sandbox`,
    run it, and return the path to the resulting pred .pkl.
    """
    src = REF_SCRIPT.read_text()

    # 1) Universal: remap the retired Ruppin_AI/st tree to its Ruppin_ST equivalent.
    #    Affects MLP weights, gene_file, super_metadata, and project_dir.
    patched = src.replace("Ruppin_AI/st/", "Ruppin_ST/")

    # 2) Make the TOP-LEVEL `from transformers import AutoModel` tolerant: the
    #    reference imports it at module top, but it's only used by foundation-
    #    model encoder branches (phikon, midnight12k, ...) we don't exercise.
    #    For the ctranspath path it's dead code, so soft-import. Anchor on a
    #    line-start so we don't break the indented nested import inside the
    #    phikon_v2 branch.
    patched = re.sub(
        r"^from transformers import AutoModel\s*$",
        "try:\n    from transformers import AutoModel\nexcept ImportError:\n    AutoModel = None",
        patched,
        flags=re.MULTILINE,
    )

    # 3) Redirect project_dir (output location) to the sandbox so we don't write
    #    into the shared Cell_revisions tree.
    patched, n = re.subn(
        r"project_dir\s*=\s*f?['\"]/vf/users/Ruppin_ST/Cell_revisions/\{project\}/['\"]",
        f"project_dir = '{sandbox}/'",
        patched,
    )
    if n == 0:
        raise RuntimeError("Could not find project_dir line to patch in reference script.")

    # The reference does `from func.utils_preprocessing import *`,
    # `from model_MLP import *`, `from utils import *`. Those modules live
    # alongside the original script in prediction_pipline/. We exec from /tmp
    # so the patched script's __file__ doesn't collide, so prepend a sys.path
    # injection that adds the original directory.
    pp_dir = REF_SCRIPT.parent
    patched = (
        f"import sys\nsys.path.insert(0, '{pp_dir}')\n"
        + patched
    )

    sandbox.mkdir(parents=True, exist_ok=True)
    cwd = REF_SCRIPT.parent
    sandbox_script = sandbox / "_1main_tcga_patched.py"
    sandbox_script.write_text(patched)

    cmd = [
        sys.executable, str(sandbox_script),
        project, str(i_slide), "ctranspath", "spot_dist",
    ]
    log = sandbox / "ref_run.log"
    print(f"running reference: {' '.join(cmd)}  (cwd={cwd}, log={log})")
    with open(log, "w") as f:
        rc = subprocess.call(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
    if rc != 0:
        print(f"reference run failed (rc={rc}). Last lines of log:")
        print(log.read_text().splitlines()[-30:])
        raise RuntimeError("reference script failed")

    # Reference writes to {sandbox}/prediction/ctranspath_spot_dist/pred/{slide_name}_pred.pkl
    pred_dir = sandbox / "prediction" / "ctranspath_spot_dist" / "pred"
    candidates = sorted(pred_dir.glob("*_pred.pkl"))
    if not candidates:
        raise RuntimeError(f"reference produced no pred .pkl in {pred_dir}")
    return candidates[0]


def run_companion(slide_path: str, slide_name: str, tile_size: int, out_dir: Path) -> Path:
    """Run the companion's predict_grid and return the pred.pkl path."""
    from path2space import Path2SpacePipeline

    pipe = Path2SpacePipeline()
    out = pipe.predict_grid(
        slide_path=slide_path,
        slide_name=slide_name,
        tile_size_px=tile_size,
        smooth=False,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / f"{slide_name}_pred.pkl"
    out["pred"].to_pickle(pred_path)
    return pred_path


def compare(ref_pkl: Path, comp_pkl: Path, atol: float) -> Tuple[bool, str]:
    ref = pd.read_pickle(ref_pkl)
    comp = pd.read_pickle(comp_pkl)

    common_idx = ref.index.intersection(comp.index)
    if not len(common_idx):
        return False, f"no overlapping tile indices ({len(ref)} vs {len(comp)})"

    gene_cols = [c for c in ref.columns if c in comp.columns and c not in ("slide_name", "x", "y", "grid_x", "grid_y")]
    if not gene_cols:
        return False, "no overlapping gene columns"

    ref_a = ref.loc[common_idx, gene_cols].values.astype(np.float64)
    comp_a = comp.loc[common_idx, gene_cols].values.astype(np.float64)

    diff = np.abs(ref_a - comp_a)
    max_abs = float(diff.max())
    mean_abs = float(diff.mean())
    ok = bool(np.allclose(ref_a, comp_a, atol=atol))

    msg = (f"compared {len(common_idx)} tiles x {len(gene_cols)} genes; "
           f"max_abs_diff={max_abs:.3e}, mean_abs_diff={mean_abs:.3e}, atol={atol:.0e}")
    return ok, msg


def main() -> int:
    ap = argparse.ArgumentParser(description="Companion-vs-reference numerical parity check.")
    ap.add_argument("--atol", type=float, default=1e-5)
    ap.add_argument("--skip-reference", action="store_true",
                    help="Reuse a previously-cached reference output (faster reruns).")
    ap.add_argument("--cache", type=Path, default=Path("/tmp/path2space_parity"),
                    help="Sandbox dir for reference outputs.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    project, i_slide, slide_name, tile_size, slide_path = pick_test_slide()
    print(f"test slide: project={project} i_slide={i_slide} name={slide_name} "
          f"tile_size={tile_size} path={slide_path}")

    args.cache.mkdir(parents=True, exist_ok=True)
    ref_pkl = args.cache / "prediction" / "ctranspath_spot_dist" / "pred" / f"{slide_name}_pred.pkl"
    if args.skip_reference and ref_pkl.exists():
        print(f"reusing cached reference output: {ref_pkl}")
    else:
        ref_pkl = patch_and_run_reference(project, i_slide, args.cache)
        print(f"reference output: {ref_pkl}")

    comp_dir = args.cache / "companion"
    comp_pkl = run_companion(slide_path, slide_name, tile_size, comp_dir)
    print(f"companion output: {comp_pkl}")

    ok, msg = compare(ref_pkl, comp_pkl, args.atol)
    print(("PASS: " if ok else "FAIL: ") + msg)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
