#!/usr/bin/env bash
#SBATCH --time=24:00:00

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OPENMDAO_REQUIRE_MPI=false
export PYTHONPATH="$REPO_ROOT/src"

EXPERIMENT_SPEC="${1:?Missing experiment spec}"
TASK_MANIFEST="${2:?Missing task manifest}"
MICROMAMBA_ENV="${3:?Missing micromamba env}"
MICROMAMBA_MODULE_USE="${4:?Missing micromamba module path}"
MICROMAMBA_MODULE="${5:?Missing micromamba module}"
TASK_INDEX="${SLURM_ARRAY_TASK_ID:?Submit with --array}"

if ! command -v module >/dev/null 2>&1; then
  source /etc/profile.d/modules.sh
fi

module use "$MICROMAMBA_MODULE_USE"
module load "$MICROMAMBA_MODULE"

exec micromamba run -n "$MICROMAMBA_ENV" \
  python -m experimenter.optimization.run \
  --experiment-file "$EXPERIMENT_SPEC" \
  --task-manifest "$TASK_MANIFEST" \
  --task-index "$TASK_INDEX"
