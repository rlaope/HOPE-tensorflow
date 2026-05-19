#!/usr/bin/env bash
# Download the HOPE / Nested Learning paper PDF from arXiv into papers/.
# The PDF is gitignored. Run this once per fresh clone.

set -euo pipefail

ARXIV_ID="2512.24695"
URL="https://arxiv.org/pdf/${ARXIV_ID}.pdf"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET_DIR="${REPO_ROOT}/papers"
TARGET="${TARGET_DIR}/nested_learning_${ARXIV_ID}.pdf"

mkdir -p "${TARGET_DIR}"

if [[ -f "${TARGET}" ]]; then
    echo "[download_paper] already present: ${TARGET}"
    exit 0
fi

echo "[download_paper] fetching ${URL}"
curl -fL --retry 3 --retry-delay 2 -o "${TARGET}" "${URL}"
echo "[download_paper] saved to ${TARGET}"
