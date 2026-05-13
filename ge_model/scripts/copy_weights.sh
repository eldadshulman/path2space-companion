#!/usr/bin/env bash
# Copy all weight artifacts into path2space-companion/weights/ so the
# companion is self-contained (no external paths at inference time).
#
# Run this ONCE, from a shell that has read access to /data/Ruppin_ST/.
# (The repo author has access; outside users will receive the populated
# weights/ directory directly.)
#
# Files copied:
#   ctranspath.pth       -> weights/ctranspath.pth                (~107 MB)
#   154 MLP checkpoints  -> weights/mlp_ensemble/result_<ik>_<il>_0/model_trained.pth
#   gene_file.pkl        -> weights/gene_file.pkl  +  weights/genes.txt
#
# Idempotent: skips files that are already present.

set -euo pipefail

COMPANION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS_DIR="${COMPANION_ROOT}/weights"
ENSEMBLE_DIR="${WEIGHTS_DIR}/mlp_ensemble"

SRC_CTRANS="${PATH2SPACE_CTRANSPATH_SRC:-/data/Ruppin_ST/11features_extraction_tai/ctranspath.pth}"
SRC_ENSEMBLE="${PATH2SPACE_ENSEMBLE_SRC:-/data/Ruppin_ST/projects/TNBC_BC_data/results}"
SRC_GENES="${PATH2SPACE_GENES_SRC:-/data/Ruppin_ST/projects/TNBC_BC_data/gene_file.pkl}"

N_IK=${PATH2SPACE_N_IK:-22}
N_IL=${PATH2SPACE_N_IL:-7}

mkdir -p "${WEIGHTS_DIR}" "${ENSEMBLE_DIR}"

echo "[1/3] ctranspath.pth"
if [[ ! -f "${WEIGHTS_DIR}/ctranspath.pth" ]]; then
    cp -v "${SRC_CTRANS}" "${WEIGHTS_DIR}/ctranspath.pth"
else
    echo "  already present, skipping"
fi

echo "[2/3] MLP ensemble (${N_IK} x ${N_IL} = $((N_IK * N_IL)) checkpoints)"
copied=0
skipped=0
for ik in $(seq 0 $((N_IK - 1))); do
    for il in $(seq 0 $((N_IL - 1))); do
        src="${SRC_ENSEMBLE}/result_${ik}_${il}_0/model_trained.pth"
        dst_dir="${ENSEMBLE_DIR}/result_${ik}_${il}_0"
        dst="${dst_dir}/model_trained.pth"
        mkdir -p "${dst_dir}"
        if [[ -f "${dst}" ]]; then
            skipped=$((skipped + 1))
            continue
        fi
        if [[ ! -f "${src}" ]]; then
            echo "  MISSING: ${src}"
            continue
        fi
        cp "${src}" "${dst}"
        copied=$((copied + 1))
    done
done
echo "  copied=${copied}, skipped=${skipped}"

echo "[3/3] gene_file.pkl + genes.txt"
if [[ ! -f "${WEIGHTS_DIR}/gene_file.pkl" ]]; then
    cp -v "${SRC_GENES}" "${WEIGHTS_DIR}/gene_file.pkl"
fi
if [[ ! -f "${WEIGHTS_DIR}/genes.txt" ]]; then
    python - <<PYEOF
import pandas as pd
df = pd.read_pickle("${WEIGHTS_DIR}/gene_file.pkl")
genes = df["gene"].astype(str).tolist()
with open("${WEIGHTS_DIR}/genes.txt", "w") as f:
    f.write("\n".join(genes) + "\n")
print(f"  wrote ${WEIGHTS_DIR}/genes.txt ({len(genes)} genes)")
PYEOF
fi

echo
echo "Done. Verify:"
echo "  ls -lh ${WEIGHTS_DIR}/ctranspath.pth"
echo "  ls ${ENSEMBLE_DIR} | wc -l   # expect $((N_IK * N_IL))"
echo "  wc -l ${WEIGHTS_DIR}/genes.txt"
