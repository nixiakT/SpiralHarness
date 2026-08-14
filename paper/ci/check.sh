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

reject_rg_match() {
  local match_message="$1"
  shift
  local scan_status
  if rg "$@"; then
    printf '%s\n' "$match_message" >&2
    exit 1
  else
    scan_status=$?
    if (( scan_status != 1 )); then
      printf 'ripgrep audit failed with status %d: %s\n' \
        "$scan_status" "$match_message" >&2
      exit 2
    fi
  fi
}

require_rg_match() {
  local missing_message="$1"
  shift
  local scan_status
  if rg "$@"; then
    return 0
  else
    scan_status=$?
    if (( scan_status == 1 )); then
      printf '%s\n' "$missing_message" >&2
      exit 1
    fi
    printf 'ripgrep audit failed with status %d: %s\n' \
      "$scan_status" "$missing_message" >&2
    exit 2
  fi
}

sha256sum --check <<'EOF'
797deef41724e93761426ac0cbcca46279a91cc650dd1f0ce76a4f08d2098ea6  iclr2027_conference.sty
2d67552db7ed38ccfccb5957b52f95656e25c249724761d3cf5f7922ad1844c5  iclr2027_conference.bst
90473c4d0542070db244cea73ef962d6cddc5b2a746757e6a40ddf5fdfb90ba9  math_commands.tex
b56ec4434b9f4607529a4b23dc68ad8d4b94f1f631c8cddaf7da78140d53a5ea  fancyhdr.sty
88bc70c0e48461934cab5b2accef06b74a8b3ac45ad03ccd3f2a6b7e0d6d530d  natbib.sty
EOF

if [[ "${SUBMISSION_READY:-0}" == "1" ]]; then
  release_attestation='../docs/iclr-2027/submission-release.yml'
  preregistration='../docs/iclr-2027/preregistered-main-study.md'
  claim_ledger='../docs/iclr-2027/claim-evidence-ledger.md'
  if [[ ! -f "$release_attestation" || ! -f "$preregistration" || \
        ! -f "$claim_ledger" ]]; then
    printf 'submission-ready build is missing research-release governance files\n' >&2
    exit 2
  fi
  require_rg_match 'release attestation is not submission-ready' \
    -q '^status:[[:space:]]+submission-ready[[:space:]]*$' "$release_attestation"
  require_rg_match 'prospective study document is not frozen' \
    -q '^Status:[[:space:]]+`frozen`\.' "$preregistration"
  require_rg_match 'release attestation lacks a frozen preregistration commit' \
    -q '^preregistration_commit:[[:space:]]+[0-9a-f]{40}[[:space:]]*$' \
    "$release_attestation"
  for digest_field in \
    preregistration_sha256 \
    claim_ledger_sha256 \
    results_bundle_sha256 \
    anonymous_supplement_sha256 \
    anonymous_supplement_audit_sha256 \
    independent_reproduction_receipt_sha256; do
    require_rg_match "release attestation lacks ${digest_field}" \
      -q "^${digest_field}:[[:space:]]+[0-9a-f]{64}[[:space:]]*$" \
      "$release_attestation"
  done
  declared_preregistration_sha256="$(
    sed -n 's/^preregistration_sha256:[[:space:]]*//p' "$release_attestation"
  )"
  actual_preregistration_sha256="$(sha256sum "$preregistration" | awk '{print $1}')"
  declared_claim_ledger_sha256="$(
    sed -n 's/^claim_ledger_sha256:[[:space:]]*//p' "$release_attestation"
  )"
  actual_claim_ledger_sha256="$(sha256sum "$claim_ledger" | awk '{print $1}')"
  if [[ "$declared_preregistration_sha256" != "$actual_preregistration_sha256" ||
        "$declared_claim_ledger_sha256" != "$actual_claim_ledger_sha256" ]]; then
    printf 'release attestation does not bind the current protocol and claim ledger\n' >&2
    exit 1
  fi
  release_marker_pattern='\\(evidenceblocked|designvalue)\b'
  release_status_pattern='pilot-pending|result-pending|sealed estimate[^}[:cntrl:]]*pending|placeholder (abstract|result|table|figure)'
  reject_rg_match 'submission-ready build still contains evidence/design markers' \
    -n -i "$release_marker_pattern" main.tex sections figures --glob '*.tex'
  reject_rg_match 'submission-ready build still contains pending or placeholder status' \
    -n -i "$release_status_pattern" main.tex sections figures --glob '*.tex'
fi

reject_rg_match 'anonymous build enables \iclrfinalcopy' \
  -n '^[[:space:]]*\\iclrfinalcopy' main.tex
format_override_pattern='\\(geometry|fontsize|linespread|setlength|addtolength|paperwidth|paperheight|textwidth|textheight|oddsidemargin|evensidemargin|topmargin|baselinestretch)'
reject_rg_match 'authored source appears to override an ICLR formatting parameter' \
  -n "$format_override_pattern" main.tex macros sections figures --glob '*.tex'

latexmk -norc -pdf -file-line-error -halt-on-error -interaction=nonstopmode main.tex

if [[ ! -s main.pdf ]]; then
  printf 'LaTeX completed without a non-empty main.pdf\n' >&2
  exit 1
fi

warning_pattern='(^! |LaTeX Warning: (Citation|Reference).*undefined|There were undefined references|There were multiply-defined labels|Label\(s\) may have changed|Package natbib Warning: Citation.*undefined|Package hyperref Warning: Token not allowed in a PDF string|Overfull \\[hv]box|Float too large for page|Missing character: There is no)'
reject_rg_match 'fatal LaTeX warning found; inspect main.log' \
  -n "$warning_pattern" main.log
if [[ -f main.blg ]]; then
  reject_rg_match 'BibTeX warning found; inspect main.blg' \
    -n '^Warning--' main.blg
fi

main_pages="$(sed -n 's/.*MGVH_MAIN_TEXT_PAGES=\([0-9][0-9]*\).*/\1/p' main.log | tail -n 1)"
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
font_report="$(mktemp)"
pdf_text_report="$(mktemp)"
trap 'rm -f "$font_report" "$pdf_text_report"' EXIT
pdffonts main.pdf > "$font_report"
reject_rg_match 'main.pdf contains Type 3 fonts' -n 'Type 3' "$font_report"
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
reject_rg_match 'source contains a local-path, credential, or private-address pattern' \
  -n -i "$anon_pattern" . --glob '*.tex' --glob '*.bib' --glob '*.md' \
  --glob '*.sh' --glob '!ci/check.sh'
pdftotext main.pdf "$pdf_text_report"
reject_rg_match 'PDF text contains a local-path, credential, or private-address pattern' \
  -n -i "$anon_pattern" "$pdf_text_report"

if [[ -n "${FORBIDDEN_ANON_REGEX:-}" ]]; then
  reject_rg_match 'source matched FORBIDDEN_ANON_REGEX' \
    -n -i "$FORBIDDEN_ANON_REGEX" . --glob '*.tex' --glob '*.bib' \
    --glob '*.md' --glob '*.sh' --glob '!ci/check.sh'
  reject_rg_match 'PDF text matched FORBIDDEN_ANON_REGEX' \
    -n -i "$FORBIDDEN_ANON_REGEX" "$pdf_text_report"
fi

if command -v qpdf >/dev/null 2>&1; then
  qpdf --check main.pdf
fi

printf 'paper checks passed: main-text pages=%s, total pages=' "$main_pages"
pdfinfo main.pdf | awk '/^Pages:/ {print $2}'
