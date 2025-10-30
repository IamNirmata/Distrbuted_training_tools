#!/usr/bin/env bash
# Optional helper to render the main report to PDF (requires pandoc + a LaTeX engine)
set -euo pipefail
cd "$(dirname "$0")"

in_md="AI Assignment-16-h100.md"
out_pdf="AI Assignment-16-h100.pdf"

if ! command -v pandoc >/dev/null 2>&1; then
  echo "pandoc not found. Install pandoc (and optionally tectonic or xelatex) and re-run." >&2
  exit 2
fi

# Choose a LaTeX engine if available
engine=""
if command -v tectonic >/dev/null 2>&1; then
  engine="--pdf-engine=tectonic"
elif command -v xelatex >/dev/null 2>&1; then
  engine="--pdf-engine=xelatex"
fi

pandoc ${engine} \
  --from gfm \
  --toc \
  --metadata title="AI Assignment 16 — Llama 3 Function Calling on 2× H100 Nodes" \
  --metadata author="Distributed Training Team" \
  --output "$out_pdf" \
  "$in_md"

echo "Wrote $out_pdf"
