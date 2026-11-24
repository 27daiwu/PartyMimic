# Sourced by build.sh / start.sh. Python/CycloneDDS environment written by build.sh (scripts/ft_env.local.sh).
_FT_SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}" SIM2REAL_ORT_NUM_THREADS="${SIM2REAL_ORT_NUM_THREADS:-2}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
unset ALL_PROXY all_proxy HTTP_PROXY http_proxy HTTPS_PROXY https_proxy
[ -f "$_FT_SCRIPTS/ft_env.local.sh" ] && . "$_FT_SCRIPTS/ft_env.local.sh"
true
