#!/usr/bin/env bash
set -euo pipefail

ENVIRONMENT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
SOURCE_ROOT="$ENVIRONMENT_ROOT/.source/gorilla"
VIRTUALENV_ROOT="$ENVIRONMENT_ROOT/.venv"
BOOTSTRAP_PYTHON="/usr/bin/python3.10"
EXCLUDE_NEWER="2026-03-24T00:00:00Z"

if [[ -z ${BFCL_UV_EXECUTABLE:-} ]]; then
  echo "set BFCL_UV_EXECUTABLE to the absolute, sealed uv 0.11.9 executable" >&2
  exit 1
fi
UV_EXECUTABLE="$BFCL_UV_EXECUTABLE"

run_bootstrap_check() {
  "$BOOTSTRAP_PYTHON" -I -S -B "$ENVIRONMENT_ROOT/verify_snapshot.py" "$@"
}
run_installed_check() {
  "$VIRTUALENV_ROOT/bin/python" -I -S -B "$ENVIRONMENT_ROOT/verify_snapshot.py" \
    installed --venv "$VIRTUALENV_ROOT"
}

run_bootstrap_check tools --uv "$UV_EXECUTABLE"
"$BOOTSTRAP_PYTHON" -I -S -B "$ENVIRONMENT_ROOT/self_test.py"
run_bootstrap_check source --root "$SOURCE_ROOT"

# The installed-tree preflight happens before either uv command so an existing
# environment is never silently repaired before inspection.
run_installed_check
"$UV_EXECUTABLE" lock \
  --check \
  --no-config \
  --exclude-newer "$EXCLUDE_NEWER" \
  --link-mode copy \
  --project "$ENVIRONMENT_ROOT" \
  --python 3.12.13 \
  --managed-python
UV_PROJECT_ENVIRONMENT="$VIRTUALENV_ROOT" "$UV_EXECUTABLE" sync \
  --check \
  --frozen \
  --no-config \
  --exclude-newer "$EXCLUDE_NEWER" \
  --link-mode copy \
  --no-dev \
  --no-install-project \
  --project "$ENVIRONMENT_ROOT" \
  --python 3.12.13 \
  --managed-python
run_installed_check

# ``env -i`` removes credentials, tokens, proxy configuration, and provider
# configuration.  The smoke still is not a syscall or network namespace.
env -i \
  LC_ALL=C.UTF-8 \
  PATH=/usr/bin:/bin \
  PYTHONDONTWRITEBYTECODE=1 \
  TZ=UTC \
  "$VIRTUALENV_ROOT/bin/python" -I -S -B "$ENVIRONMENT_ROOT/smoke_evaluator.py" \
  --venv "$VIRTUALENV_ROOT"
run_installed_check
