#!/usr/bin/env bash
# PartyMimic Pipeline V1.0 entry point. Usage: ./pm <prepare|preview|train|status|resume|stop|eval|export|sim2sim|package|analyze|run|doctor> <dance> [options]   (./pm --help)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/.local/bin:$PATH" PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$HERE/pipeline:$HERE/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$HERE/.venv/bin/python" -m pmpipe.cli "$@"
