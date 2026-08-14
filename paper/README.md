# ICLR 2027 paper source

Status: **working, anonymous, evidence-blocked draft**. It is structurally
compatible with the official ICLR 2027 LaTeX template, but it is not ready for
submission because the confirmatory experiment and independent artifact audit
have not finished.

## Official template provenance

The official archive was obtained from:

- <https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip>
- SHA-256 of the downloaded ZIP:
  `0d940dfa9398ae99a18f24a85a8a683f367204b6af6d17d2899e60a67102529e`
- Provenance checked: 2026-08-14.

The following files were copied byte-for-byte from the archive. They must not
be formatted, patched, or otherwise edited:

| File | SHA-256 |
|---|---|
| `iclr2027_conference.sty` | `797deef41724e93761426ac0cbcca46279a91cc650dd1f0ce76a4f08d2098ea6` |
| `iclr2027_conference.bst` | `2d67552db7ed38ccfccb5957b52f95656e25c249724761d3cf5f7922ad1844c5` |
| `math_commands.tex` | `90473c4d0542070db244cea73ef962d6cddc5b2a746757e6a40ddf5fdfb90ba9` |
| `fancyhdr.sty` | `b56ec4434b9f4607529a4b23dc68ad8d4b94f1f631c8cddaf7da78140d53a5ea` |
| `natbib.sty` | `88bc70c0e48461934cab5b2accef06b74a8b3ac45ad03ccd3f2a6b7e0d6d530d` |

The local `fancyhdr.sty` and `natbib.sty` copies are intentional transitive
dependencies from the official archive: `iclr2027_conference.sty` explicitly
requires both packages (and loads `fancyhdr` a second time harmlessly). Keeping
the archive copies beside the paper prevents a host TeX installation from
silently substituting different versions. They are not custom style changes.

Verify their integrity from `paper/` with:

```bash
sha256sum iclr2027_conference.sty iclr2027_conference.bst \
  math_commands.tex fancyhdr.sty natbib.sty
```

The source keeps `\iclrfinalcopy` commented, provides no author identity, and
sets anonymous PDF metadata. Re-download and re-verify the official archive 72
hours and 24 hours before the deadline in case the conference updates it.

## Layout

- `main.tex` is the only top-level manuscript entry point.
- `macros/notation.tex` defines notation and the visible evidence-state macros.
- `sections/` contains the paper body, required/recommended statements, and
  appendix.
- `figures/mechanism_boundary.tex` is a pure TikZ, grayscale-readable mechanism
  and authority-boundary figure.
- `references/references.bib` contains cited sources.

The red `\evidenceblocked{...}` macro is intentional. Never replace a marker
with an estimate unless the claim is bound to the frozen preregistration, all
search closures, the atomic sealed release, the prespecified analysis, and a
reproducible table or figure source. `\pilotonly{...}` marks development
evidence that cannot enter confirmatory inference, while `\designvalue{...}`
marks a preregistered design choice rather than an observed result.

Before declaring the source submission-ready, this command must return no
matches:

```bash
rg -n '\\evidenceblocked\{' main.tex sections figures --glob '*.tex'
```

USPTO and the single Penguin public task may appear only as motivating pilots.
The adaptively tuned Symptom2Disease result is excluded from the primary table
and headline claims.

## Build

With a current TeX Live installation:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

From the repository root, run the full local/CI gate (official-file hashes,
compilation, citations, warnings, nine-page main-text limit, US Letter size,
fonts, and optional submission-readiness checks) after installing TeX Live plus
`poppler-utils`:

```bash
paper/ci/check.sh
```

On Debian-family runners, the required capabilities are normally provided by
`latexmk`, `texlive-latex-base`, `texlive-latex-extra`,
`texlive-fonts-recommended`, `texlive-pictures`, `poppler-utils`, and
`ripgrep`. Freeze the runner image rather than installing unpinned packages in
the archival job.

Set `SUBMISSION_READY=1` only for a release candidate; that additionally fails
if any `\evidenceblocked` marker remains. A project-specific anonymous-name or
institution regex can be supplied without committing identities:

```bash
FORBIDDEN_ANON_REGEX='private-name|private-institution' \
  SUBMISSION_READY=1 paper/ci/check.sh
```

Equivalent manual build:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Clean generated files with `latexmk -c`; do not commit auxiliary files. The
host did not provide TeX Live or a container runtime. As a non-authoritative
smoke check on 2026-08-14, Tectonic 0.17.0's x86-64 Linux musl archive
(`sha256:8533d07f9ccbd7a65824b9e0459041bca34af1eb33daba48f59215593753a3b7`)
was downloaded to `/tmp` and compiled the anonymous source successfully: the log marker reported
8 main-text pages, the PDF had 10 total US-Letter pages, all fonts were
embedded and non-Type-3, and the metadata author was `Anonymous Authors`.
Only underfull-box and XeTeX font-substitution warnings remained. This does not
replace the required pdfLaTeX/TeX Live gate in `paper/ci/check.sh`, which must
still run in CI or a TeX-equipped environment. In hosted CI, use an
organization-approved TeX Live image pinned by immutable digest (or a LaTeX
action pinned to a full commit SHA), mount only the anonymous checkout, and run
`paper/ci/check.sh`; do not use a mutable `latest` image for the archival build.

## Pre-submission audit

1. Confirm US Letter output and an initial-submission main-text limit of nine
   pages; references and appendices may follow. `main.tex` flushes floats and
   emits `SPIRALHARNESS_MAIN_TEXT_PAGES=<n>` immediately before the statements,
   and the CI script enforces `n <= 9`. Keep the AI use, ethics, and
   reproducibility statements before references as required/recommended by the
   official shell.
2. Inspect the log for undefined references/citations, overfull boxes, missing
   glyphs, duplicate labels, and PDF-string warnings. View the figure at 100%
   and in grayscale.
3. Recompute every result from immutable aggregate inputs in a clean anonymous
   checkout. Ensure all configured conditions and failed calls are present.
4. Search the source, bibliography, PDF metadata, and supplemental archive for
   author names, institutions, usernames, local paths, API endpoints, secrets,
   repository ownership, acknowledgments, and identifying URLs.
5. Manually validate every bibliography record against its primary source and
   complete the human review commitments in the AI use statement.

No generated PDF, empirical number, or successful unit test by itself removes
an evidence block.
