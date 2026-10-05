#!/usr/bin/env bash
# Opt-in git pre-push hook: runs the push gate (`just pre-push`) inside the dev shell, in the checkout being
# pushed from. Install it with `just install-pre-push-hook`. A failing gate blocks the push. Set
# POOT_CARGO_WRAPPER to a command that should run the gate (for example a shared-build-slot wrapper).
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
if [[ -z "${IN_NIX_SHELL:-}" ]]; then
  # shellcheck disable=SC2086 # the wrapper is a command line, split on purpose
  exec ${POOT_CARGO_WRAPPER:-} nix develop -c just pre-push
fi
exec just pre-push
