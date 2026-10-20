#!/usr/bin/env bash
# Sync the pi0.5 subtask+FAST code from this development host (ylhp-e-ai) to the
# GPU training host, and optionally fetch run artefacts back.
#
# Run this ON ylhp-e-ai (it holds the ssh key for the training host):
#     bash scripts/architecture/sync_to_training_host.sh            # dry run
#     bash scripts/architecture/sync_to_training_host.sh --apply    # push code
#     bash scripts/architecture/sync_to_training_host.sh --fetch    # pull results
#
# Never uses --delete: the training host keeps its own runs/ and checkpoints/.
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}
DEST_HOST=${DEST_HOST:?Set DEST_HOST to your SSH host}
DEST_ROOT=${DEST_ROOT:?Set DEST_ROOT to the remote code directory}
ASSET_DEST=${ASSET_DEST:-$DEST_ROOT/assets}

# Code that must match between hosts. Experiments/checkpoints stay host-local.
PUSH_PATHS=(fluxvla configs scripts tests docs pyproject.toml)
FETCH_PATHS=(experiments/pi05_subtask_architecture)

SSH_OPTS=(-e "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o LogLevel=ERROR")
MODE=dry-run
case "${1:---dry-run}" in
  --apply) MODE=apply ;;
  --fetch) MODE=fetch ;;
  --dry-run|"") MODE=dry-run ;;
  -h|--help) sed -n '1,20p' "$0"; exit 0 ;;
  *) echo "unknown option: $1" >&2; exit 2 ;;
esac

echo "repo      : $REPO"
echo "training  : $DEST_HOST:$DEST_ROOT"
echo "mode      : $MODE"

if [[ "$MODE" == "fetch" ]]; then
  for path in "${FETCH_PATHS[@]}"; do
    echo "--- pulling $path"
    mkdir -p "$REPO/$(dirname "$path")"
    rsync -az --info=stats2 "${SSH_OPTS[@]}" \
      "$DEST_HOST:$DEST_ROOT/$path/" "$REPO/$path/"
  done
  echo "fetched results into $REPO"
  exit 0
fi

RSYNC_FLAGS=(-rlptz --exclude '__pycache__/' --exclude '*.pyc' --exclude '.pytest_cache/')
if [[ "$MODE" == "dry-run" ]]; then
  RSYNC_FLAGS+=(--dry-run --itemize-changes)
fi

for path in "${PUSH_PATHS[@]}"; do
  if [[ ! -e "$REPO/$path" ]]; then
    echo "skip missing: $path"
    continue
  fi
  echo "--- pushing $path"
  rsync "${RSYNC_FLAGS[@]}" "${SSH_OPTS[@]}" "$REPO/$path" "$DEST_HOST:$DEST_ROOT/"
done

if [[ "$MODE" == "apply" ]]; then
  echo "--- pushing FAST tokenizer assets"
  rsync "${RSYNC_FLAGS[@]}" "${SSH_OPTS[@]}" "$REPO/assets/fast" "$DEST_HOST:$ASSET_DEST/"
fi

# Verify a couple of files byte-for-byte instead of trusting rsync's exit code.
if [[ "$MODE" == "apply" ]]; then
  echo "--- verification"
  for file in fluxvla/models/vlas/pi05_subtask.py fluxvla/transforms/subtask_fast_prompts.py \
              configs/pi05_subtask_fast.py tests/test_subtask_fast_model.py; do
    local_sum=$(sha256sum "$REPO/$file" | cut -d' ' -f1)
    remote_sum=$(ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o LogLevel=ERROR \
      "$DEST_HOST" "sha256sum $DEST_ROOT/$file 2>/dev/null | cut -d' ' -f1" || echo missing)
    if [[ "$local_sum" == "$remote_sum" ]]; then
      echo "ok   $file"
    else
      echo "DIFF $file (local=$local_sum remote=$remote_sum)" >&2
      exit 1
    fi
  done
  echo "--- FAST tokenizer on training host"
  ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o LogLevel=ERROR "$DEST_HOST" \
    "ls -l $ASSET_DEST/fast | head -8"
fi
