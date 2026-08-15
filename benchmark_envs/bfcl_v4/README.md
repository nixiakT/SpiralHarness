# BFCL V4 public-development environment

Status: **development infrastructure only; formal exact-environment evidence is
BLOCKED**.

This directory keeps BFCL's large dependency graph outside the SpiralHarness
core wheel and provides fail-closed drift checks for a public BFCL V4 pilot. It
does not configure or call a model. It is not a hidden benchmark environment,
a signed supply-chain attestation, a network namespace, or a score-bearing
execution receipt.

Every emitted JSON observation deliberately contains:

```json
{
  "dependency_environment_attested": false,
  "network_isolation_attested": false,
  "reportable_result": false
}
```

## What is content-bound

The checked-in snapshot and verifier bind these narrow identities:

- the selected BFCL project archive from `ShishirPatil/gorilla` commit
  `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`;
- all 193 files in that selected archive and all 183 files in its
  `bfcl_eval` package tree;
- the `bfcl-eval==2026.3.23` wheel and sdist identities;
- byte equality between the release wheel's 183 package files and the package
  tree at the pinned commit;
- the checked-in `uv.lock`, `pyproject.toml`, `.python-version`, and executable
  control scripts, including their file modes;
- observations of one native uv 0.11.9 executable, one bootstrap Python
  executable, and one CPython 3.12.13 executable.

The common BFCL package-tree identity is:

```text
file count: 183
total bytes: 13,513,439
tree sha256: 3753addd78c10a6e59e3488ffdc5fb38cb46929380925ccfaecfcdb1d8b533b2
```

These are local drift controls. The manifest and verifier live in the same Git
tree and are not an independent trust root. A release artifact or repository
commit outside this directory must bind them for provenance.

## What is not attested

`uv.lock` contains 142 package-resolution records. That number must not be
described as 142 byte-exact installed packages: it includes candidate wheels
for multiple platforms and source distributions. On the audited Linux host,
uv reports 139 installed distributions, but this directory does not yet bind a
complete installed-file manifest for those distributions.

The first local build compiled `google-search-results` from an sdist. A locked
sdist hash does not by itself bind its build frontend, backend dependencies,
compiler, linker, or resulting wheel bytes. There is no complete local
wheelhouse with verified hashes. Other installed distributions are checked for
uv synchronization, not individually byte-hashed.

The observed CPython executable is hashed, but its standard library,
`lib-dynload`, shared libraries, glibc, dynamic loader, kernel, filesystem,
locale data, and container or machine image are not completely bound. The uv
binary is observed by hash and version, but its publisher provenance is not
attested. `--link-mode copy` is explicitly requested for new installs and
checks, but a read-only check cannot prove the historical link mode of an
already-created virtual environment.

Consequently:

- `dependency_environment_attested=false`;
- `full_cpython_runtime_attested=false`;
- `container_image_attested=false`; and
- BFCL output from this directory remains public/development and nonreportable.

Closing this blocker requires at least a complete platform-specific wheelhouse,
hashes for locally built wheels, a full installed-tree manifest, and a bound
container or machine image. Those artifacts are intentionally not fabricated
in this batch.

## Setup

Supply the absolute canonical path to the native uv executable whose observed
hash is recorded in `snapshot.json`:

```bash
BFCL_UV_EXECUTABLE=/absolute/path/to/uv \
  ./benchmark_envs/bfcl_v4/setup.sh
```

The setup sequence is fail-closed:

1. `/usr/bin/python3.10 -I -S -B` checks the control files and observed uv and
   bootstrap-Python executables before uv runs. `-S` prevents site startup and
   `.pth` execution.
2. A new source is fetched at the exact commit into a staging directory. A
   selected-path `git archive` is verified before its directory is moved into
   place. Existing source is checked read-only and never repaired.
3. `uv lock --check` runs with explicit `--no-config`, upload cutoff, Python
   version, and `--link-mode copy` arguments.
4. If `.venv` already exists, the venv Python runs the installed-tree preflight
   with `-I -S -B` **before** `uv sync --check`. No mutating sync is allowed to
   repair an existing environment. Every sync explicitly fixes
   `UV_PROJECT_ENVIRONMENT` to this directory's `.venv`, so an inherited value
   cannot redirect the check.
5. Only a genuinely absent `.venv` may be populated, using `uv sync --frozen
   --no-config --link-mode copy`. A build-in-progress marker remains after a
   failed build so a later setup refuses silent repair.

Virtual environments cannot safely be created under a temporary name and then
renamed because installed console-script shebangs can embed the original path.
The build marker is therefore used instead of pretending the venv install is
atomic.

The source archive and venv live under ignored `.source/` and `.venv/`
directories. Existence guards use both `-e` and `-L`, so dangling symlinks are
refused rather than treated as absent or overwritten. If either path fails
preflight, quarantine it and deliberately rebuild; the scripts do not delete or
overwrite it.

## Read-only verification

```bash
BFCL_UV_EXECUTABLE=/absolute/path/to/uv \
  ./benchmark_envs/bfcl_v4/verify.sh
```

Verification performs, in order:

1. observed-tool and control-file checks plus seven lightweight negative and
   control-contract verifier self-tests;
2. source-tree verification;
3. installed-tree and no-bytecode preflight under venv Python `-I -S -B`;
4. explicit `uv lock --check` and `uv sync --check` with `--no-config`,
   `--link-mode copy`, and `UV_PROJECT_ENVIRONMENT` fixed to this environment;
5. a second installed-tree check;
6. one public `agentic_checker` positive/negative smoke under `env -i` with
   only locale, path, timezone, and no-bytecode variables; and
7. a final installed-tree/no-bytecode check.

The smoke clears credential, token, provider, and proxy variables by starting
from an empty environment. Its Python audit hook is only a narrow in-process
tripwire for several documented socket events. It is not complete network
observation and cannot block direct syscalls, native extensions, subprocesses,
shared-memory helpers, or another process. The smoke therefore reports the
literal observations:

```json
{
  "audited_socket_events_observed": [],
  "provider_calls_requested": 0,
  "network_isolation_attested": false,
  "syscall_sandbox_attested": false
}
```

Formal network isolation would require a separately verified namespace,
container, firewall, or syscall policy plus an execution receipt.

## Filesystem and race boundary

The verifier canonicalizes its own environment root, requires source and venv
paths to remain contained below it, rejects symlink components below that root,
rejects symlinks within verified BFCL trees, and checks containment again while
enumerating files. The `/home` path on the audited host is itself an alias, so
scripts use the canonical physical root returned by `pwd -P`.

There remains a time-of-check/time-of-use window between metadata checks,
content reads, uv checks, and the later import. A concurrent process with write
access can race those operations. Mount replacement and host-level compromise
are also outside this verifier. A formal run needs a read-only content-addressed
filesystem or image and a trusted launcher.

## Bytecode cleanup record

Red-team review created exactly two generated cache directories before the
`-S` hardening:

- `.venv/lib/python3.12/site-packages/__pycache__`;
- `.venv/lib/python3.12/site-packages/_distutils_hack/__pycache__`.

With no validation process running, both generated directories were removed
from the venv and quarantined at
`/tmp/bfcl-reviewer-bytecode.OUTBNXJB`. No source or installed package data was
deleted. Subsequent `-I -S -B` preflight, smoke, and postflight checks observed
no project-venv or BFCL-source bytecode.

The uv-managed base interpreter can contain vendor-generated standard-library
bytecode outside `.venv`; that external runtime tree is not covered by the
no-bytecode check or claimed as attested.

## Evidence boundary

The checks establish that one local public-development installation presently
matches the narrow BFCL package/source identities and passes a tiny evaluator
import smoke. They do not attest a provider, inference request, task runtime,
full BFCL evaluation, score, hidden split, or causal experiment. Public BFCL
questions, answers, and grader remain public.

No inference credential or inference endpoint belongs in this directory, its
source archive, logs, or receipts.
