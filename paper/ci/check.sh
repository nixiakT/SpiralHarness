#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=C
paper_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$paper_dir"

for command_name in sha256sum latexmk pdflatex bibtex rg pdfinfo pdffonts pdftotext; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    printf 'missing required command: %s\n' "$command_name" >&2
    exit 2
  fi
done

sha256sum --check <<'EOF'
797deef41724e93761426ac0cbcca46279a91cc650dd1f0ce76a4f08d2098ea6  iclr2027_conference.sty
2d67552db7ed38ccfccb5957b52f95656e25c249724761d3cf5f7922ad1844c5  iclr2027_conference.bst
90473c4d0542070db244cea73ef962d6cddc5b2a746757e6a40ddf5fdfb90ba9  math_commands.tex
b56ec4434b9f4607529a4b23dc68ad8d4b94f1f631c8cddaf7da78140d53a5ea  fancyhdr.sty
88bc70c0e48461934cab5b2accef06b74a8b3ac45ad03ccd3f2a6b7e0d6d530d  natbib.sty
EOF

if [[ "${SUBMISSION_READY:-0}" == "1" ]] &&
   rg -n '\\evidenceblocked\{' main.tex sections figures --glob '*.tex'; then
  printf 'submission-ready build still contains evidence blockers\n' >&2
  exit 1
fi

if rg -n '^[[:space:]]*\\iclrfinalcopy' main.tex; then
  printf 'anonymous build enables \\iclrfinalcopy\n' >&2
  exit 1
fi
format_override_pattern='\\(geometry|fontsize|linespread|setlength|addtolength|paperwidth|paperheight|textwidth|textheight|oddsidemargin|evensidemargin|topmargin|baselinestretch)'
if rg -n "$format_override_pattern" main.tex macros sections figures \
     --glob '*.tex'; then
  printf 'authored source appears to override an ICLR formatting parameter\n' >&2
  exit 1
fi

latexmk -norc -pdf -file-line-error -halt-on-error -interaction=nonstopmode main.tex

if [[ ! -s main.pdf ]]; then
  printf 'LaTeX completed without a non-empty main.pdf\n' >&2
  exit 1
fi

warning_pattern='(^! |LaTeX Warning: (Citation|Reference).*undefined|There were undefined references|There were multiply-defined labels|Label\(s\) may have changed|Package natbib Warning: Citation.*undefined|Package hyperref Warning: Token not allowed in a PDF string|Overfull \\[hv]box|Float too large for page|Missing character: There is no)'
if rg -n "$warning_pattern" main.log; then
  printf 'fatal LaTeX warning found; inspect main.log\n' >&2
  exit 1
fi
if [[ -f main.blg ]] && rg -n '^Warning--' main.blg; then
  printf 'BibTeX warning found; inspect main.blg\n' >&2
  exit 1
fi

main_pages="$(sed -n 's/.*SPIRALHARNESS_MAIN_TEXT_PAGES=\([0-9][0-9]*\).*/\1/p' main.log | tail -n 1)"
if [[ -z "$main_pages" ]]; then
  printf 'main-text page marker missing from main.log\n' >&2
  exit 1
fi
if (( main_pages > 9 )); then
  printf 'main text has %d pages; ICLR 2027 initial limit is 9\n' "$main_pages" >&2
  exit 1
fi

if ! pdfinfo main.pdf | rg -q '^Page size:[[:space:]]+612 x 792 pts'; then
  printf 'main.pdf is not US Letter (612 x 792 pt)\n' >&2
  pdfinfo main.pdf | rg '^Page size:' >&2 || true
  exit 1
fi
if ! pdfinfo main.pdf | rg -q '^Author:[[:space:]]+Anonymous Authors[[:space:]]*$'; then
  printf 'main.pdf does not have anonymous author metadata\n' >&2
  pdfinfo main.pdf | rg '^(Author|Creator|Producer):' >&2 || true
  exit 1
fi
if pdffonts main.pdf | rg -q 'Type 3'; then
  printf 'main.pdf contains Type 3 fonts\n' >&2
  pdffonts main.pdf >&2
  exit 1
fi

font_report="$(mktemp)"
trap 'rm -f "$font_report"' EXIT
pdffonts main.pdf > "$font_report"
if awk '
  NR == 1 { embedded_column = index($0, "emb"); next }
  NR <= 2 || NF == 0 { next }
  substr($0, embedded_column, 3) != "yes" { bad = 1 }
  END { exit bad ? 0 : 1 }
' "$font_report"; then
  printf 'main.pdf contains an unembedded font\n' >&2
  cat "$font_report" >&2
  exit 1
fi

anon_pattern='/(home|Users)/[^/[:space:]]+|sk-[A-Za-z0-9]{16,}|(10|127|169\.254|192\.168)\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}|172\.(1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}\.[0-9]{1,3}'
if rg -n -i "$anon_pattern" . \
     --glob '*.tex' --glob '*.bib' --glob '*.md' --glob '*.sh' \
     --glob '!ci/check.sh'; then
  printf 'source contains a local-path, credential, or private-address pattern\n' >&2
  exit 1
fi
if pdftotext main.pdf - | rg -n -i "$anon_pattern"; then
  printf 'PDF text contains a local-path, credential, or private-address pattern\n' >&2
  exit 1
fi

if [[ -n "${FORBIDDEN_ANON_REGEX:-}" ]]; then
  if rg -n -i "$FORBIDDEN_ANON_REGEX" . \
       --glob '*.tex' --glob '*.bib' --glob '*.md' --glob '*.sh' \
       --glob '!ci/check.sh'; then
    printf 'source matched FORBIDDEN_ANON_REGEX\n' >&2
    exit 1
  fi
  if pdftotext main.pdf - | rg -n -i "$FORBIDDEN_ANON_REGEX"; then
    printf 'PDF text matched FORBIDDEN_ANON_REGEX\n' >&2
    exit 1
  fi
fi

if command -v qpdf >/dev/null 2>&1; then
  qpdf --check main.pdf
fi

printf 'paper checks passed: main-text pages=%s, total pages=' "$main_pages"
pdfinfo main.pdf | awk '/^Pages:/ {print $2}'
