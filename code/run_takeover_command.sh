#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export SSR0="$TASK_ROOT"
# env.sh supports unset legacy environment variables when sourced without nounset.
set +u
source "$TASK_ROOT/env.sh"
set -u
cd "$TASK_ROOT"
exec "$@"
