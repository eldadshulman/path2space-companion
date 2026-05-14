#!/usr/bin/env bash
# Populate path2space-companion/ge_model/weights/ with the trained weights.
#
# Two sources, tried in order:
#
#   1. Zenodo (public; default for outside users)
#      Fetches ctranspath.pth + mlp_ensemble.tar.gz + genes.txt + MD5SUMS.txt
#      from the public deposit at https://doi.org/10.5281/zenodo.20174301,
#      verifies MD5 sums, and extracts the ensemble in place. Resumable.
#
#   2. Lab-internal copy (Ruppin Lab members)
#      Used when the source paths under /data/Ruppin_ST/ are reachable and
#      `PATH2SPACE_USE_ZENODO=1` is not set. Faster on the cluster.
#
# Force Zenodo:    PATH2SPACE_USE_ZENODO=1 bash scripts/copy_weights.sh
# Force internal:  PATH2SPACE_USE_INTERNAL=1 bash scripts/copy_weights.sh
#
# Idempotent: skips files that are already present.

set -euo pipefail

COMPANION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WEIGHTS_DIR="${COMPANION_ROOT}/weights"
ENSEMBLE_DIR="${WEIGHTS_DIR}/mlp_ensemble"

ZENODO_RECORD="${PATH2SPACE_ZENODO_RECORD:-20174301}"
ZENODO_BASE="https://zenodo.org/records/${ZENODO_RECORD}/files"

SRC_CTRANS="${PATH2SPACE_CTRANSPATH_SRC:-/data/Ruppin_ST/11features_extraction_tai/ctranspath.pth}"
SRC_ENSEMBLE="${PATH2SPACE_ENSEMBLE_SRC:-/data/Ruppin_ST/projects/TNBC_BC_data/results}"
SRC_GENES="${PATH2SPACE_GENES_SRC:-/data/Ruppin_ST/projects/TNBC_BC_data/gene_file.pkl}"

N_IK=${PATH2SPACE_N_IK:-22}
N_IL=${PATH2SPACE_N_IL:-7}

mkdir -p "${WEIGHTS_DIR}" "${ENSEMBLE_DIR}"

# -------- Pick a source ----------------------------------------------------

source_kind=""
if [[ "${PATH2SPACE_USE_INTERNAL:-0}" == "1" ]]; then
    source_kind="internal"
elif [[ "${PATH2SPACE_USE_ZENODO:-0}" == "1" ]]; then
    source_kind="zenodo"
elif [[ -f "${SRC_CTRANS}" && -d "${SRC_ENSEMBLE}" && -f "${SRC_GENES}" ]]; then
    source_kind="internal"
else
    source_kind="zenodo"
fi
echo "Source: ${source_kind}"

# -------- Zenodo download path --------------------------------------------

zenodo_fetch() {
    local n_expected=$((N_IK * N_IL))
    local n_have=0
    if [[ -d "${ENSEMBLE_DIR}" ]]; then
        n_have=$(find "${ENSEMBLE_DIR}" -name model_trained.pth 2>/dev/null | wc -l)
    fi
    if [[ -f "${WEIGHTS_DIR}/ctranspath.pth" && -f "${WEIGHTS_DIR}/genes.txt" && "${n_have}" -eq "${n_expected}" ]]; then
        echo "  weights already populated; skipping Zenodo fetch"
        return 0
    fi

    if ! command -v wget >/dev/null 2>&1; then
        echo "  ERROR: wget not found; install wget or use --use-internal" >&2
        exit 1
    fi

    cd "${WEIGHTS_DIR}"

    echo "[Zenodo 1/4] ctranspath.pth"
    [[ -f ctranspath.pth ]] || wget -c "${ZENODO_BASE}/ctranspath.pth"

    echo "[Zenodo 2/4] genes.txt"
    [[ -f genes.txt ]] || wget -c "${ZENODO_BASE}/genes.txt"

    echo "[Zenodo 3/4] mlp_ensemble.tar.gz (~6.1 GB)"
    if [[ "${n_have}" -ne "${n_expected}" ]]; then
        [[ -f mlp_ensemble.tar.gz ]] || wget -c "${ZENODO_BASE}/mlp_ensemble.tar.gz"
    fi

    echo "[Zenodo 4/4] MD5SUMS.txt + verify"
    wget -q -O MD5SUMS.txt "${ZENODO_BASE}/MD5SUMS.txt"
    md5sum -c MD5SUMS.txt

    if [[ -f mlp_ensemble.tar.gz ]]; then
        echo "[Zenodo] extracting mlp_ensemble.tar.gz"
        tar -xzf mlp_ensemble.tar.gz
        rm -f mlp_ensemble.tar.gz
    fi

    cd - >/dev/null
    echo "Zenodo fetch done."
}

# -------- Lab-internal copy path ------------------------------------------

internal_copy() {
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
}

# -------- Run -------------------------------------------------------------

if [[ "${source_kind}" == "zenodo" ]]; then
    zenodo_fetch
else
    internal_copy
fi

echo
echo "Done. Verify:"
echo "  ls -lh ${WEIGHTS_DIR}/ctranspath.pth"
echo "  find ${ENSEMBLE_DIR} -name model_trained.pth | wc -l   # expect $((N_IK * N_IL))"
echo "  wc -l ${WEIGHTS_DIR}/genes.txt"
