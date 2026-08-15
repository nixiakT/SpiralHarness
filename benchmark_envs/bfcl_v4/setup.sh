#!/usr/bin/env bash
set -euo pipefail

ENVIRONMENT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
SOURCE_ROOT="$ENVIRONMENT_ROOT/.source/gorilla"
VIRTUALENV_ROOT="$ENVIRONMENT_ROOT/.venv"
BUILD_MARKER="$ENVIRONMENT_ROOT/.venv-build-in-progress"
BOOTSTRAP_PYTHON="/usr/bin/python3.10"
UPSTREAM_REPOSITORY="https://github.com/ShishirPatil/gorilla.git"
UPSTREAM_COMMIT="6ea57973c7a6097fd7c5915698c54c17c5b1b6c8"
EXCLUDE_NEWER="2026-03-24T00:00:00Z"

if [[ -z ${BFCL_UV_EXECUTABLE:-} ]]; then
  echo "set BFCL_UV_EXECUTABLE to the absolute, sealed uv 0.11.9 executable" >&2
  exit 1
fi
UV_EXECUTABLE="$BFCL_UV_EXECUTABLE"

if [[ $(uname -s) != "Linux" || $(uname -m) != "x86_64" ]]; then
  echo "BFCL development environment is limited to Linux x86_64" >&2
  exit 1
fi
if [[ ! -x $BOOTSTRAP_PYTHON ]]; then
  echo "sealed bootstrap Python is unavailable: $BOOTSTRAP_PYTHON" >&2
  exit 1
fi

stage_roots=()
cleanup_stages() {
  local stage_root
  for stage_root in "${stage_roots[@]}"; do
    case "$stage_root" in
      "$ENVIRONMENT_ROOT"/.source-stage.*) rm -rf -- "$stage_root" ;;
      *) echo "refusing to clean unexpected stage path: $stage_root" >&2 ;;
    esac
  done
}
trap cleanup_stages EXIT

run_bootstrap_check() {
  "$BOOTSTRAP_PYTHON" -I -S -B "$ENVIRONMENT_ROOT/verify_snapshot.py" "$@"
}
run_installed_check() {
  local venv_root=$1
  "$venv_root/bin/python" -I -S -B "$ENVIRONMENT_ROOT/verify_snapshot.py" \
    installed --venv "$venv_root"
}
path_exists_or_symlink() {
  local candidate=$1
  [[ -e $candidate || -L $candidate ]]
}
run_lock_check() {
  "$UV_EXECUTABLE" lock \
    --check \
    --no-config \
    --exclude-newer "$EXCLUDE_NEWER" \
    --link-mode copy \
    --project "$ENVIRONMENT_ROOT" \
    --python 3.12.13 \
    --managed-python
}
run_sync_check() {
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
}

# The sealed native uv observation is checked before uv executes.
run_bootstrap_check tools --uv "$UV_EXECUTABLE"
"$BOOTSTRAP_PYTHON" -I -S -B "$ENVIRONMENT_ROOT/self_test.py"

if path_exists_or_symlink "$SOURCE_ROOT"; then
  run_bootstrap_check source --root "$SOURCE_ROOT"
else
  if path_exists_or_symlink "$ENVIRONMENT_ROOT/.source"; then
    echo "refusing to repair incomplete source directory: $ENVIRONMENT_ROOT/.source" >&2
    exit 1
  fi
  source_stage=$(mktemp -d "$ENVIRONMENT_ROOT/.source-stage.XXXXXXXX")
  stage_roots+=("$source_stage")
  git init --quiet "$source_stage/repository"
  git -C "$source_stage/repository" remote add origin "$UPSTREAM_REPOSITORY"
  git -C "$source_stage/repository" fetch \
    --quiet \
    --depth=1 \
    --filter=blob:none \
    origin "$UPSTREAM_COMMIT"
  fetched_commit=$(git -C "$source_stage/repository" rev-parse FETCH_HEAD)
  if [[ $fetched_commit != "$UPSTREAM_COMMIT" ]]; then
    echo "fetched BFCL commit differs: $fetched_commit" >&2
    exit 1
  fi

  mkdir -p "$source_stage/final-source/gorilla"
  git -C "$source_stage/repository" archive \
    --format=tar \
    "$UPSTREAM_COMMIT" \
    -- LICENSE berkeley-function-call-leaderboard \
    | tar --extract --file=- --directory="$source_stage/final-source/gorilla"
  run_bootstrap_check source --root "$source_stage/final-source/gorilla"
  if path_exists_or_symlink "$ENVIRONMENT_ROOT/.source"; then
    echo "source destination appeared during staging; refusing replacement" >&2
    exit 1
  fi
  mv -- "$source_stage/final-source" "$ENVIRONMENT_ROOT/.source"
fi

run_lock_check

if path_exists_or_symlink "$VIRTUALENV_ROOT"; then
  if path_exists_or_symlink "$BUILD_MARKER"; then
    echo "existing virtualenv has an unfinished-build marker; refusing repair" >&2
    exit 1
  fi
  # Existing environments receive only read-only preflight checks.  In
  # particular, sync is never allowed to repair them before integrity checks.
  run_installed_check "$VIRTUALENV_ROOT"
  run_sync_check
else
  if path_exists_or_symlink "$BUILD_MARKER"; then
    echo "unfinished virtualenv build marker exists; inspect it manually" >&2
    exit 1
  fi
  mkdir "$BUILD_MARKER"
  UV_PROJECT_ENVIRONMENT="$VIRTUALENV_ROOT" "$UV_EXECUTABLE" sync \
    --frozen \
    --no-config \
    --exclude-newer "$EXCLUDE_NEWER" \
    --link-mode copy \
    --no-dev \
    --no-install-project \
    --project "$ENVIRONMENT_ROOT" \
    --python 3.12.13 \
    --managed-python
  run_installed_check "$VIRTUALENV_ROOT"
  rmdir "$BUILD_MARKER"
fi

BFCL_UV_EXECUTABLE="$UV_EXECUTABLE" "$ENVIRONMENT_ROOT/verify.sh"
