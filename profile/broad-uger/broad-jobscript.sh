#!/usr/bin/env bash
# properties = {properties}
# Snakemake formats only properties and exec_job; shell braces are doubled.
set -e
if [[ -r /broad/software/scripts/useuse ]]; then
    # Dotkit returns non-zero inside `use`: under set -e the job would end there with status 8 (W-767).
    set +e
    source /broad/software/scripts/useuse
    use UGER
    set -e
fi

: "${{STARFINDER_IMAGE:?set by broad-submit.py}}"
: "${{STARFINDER_SIF:?set by broad-submit.py}}"
: "${{STARFINDER_THREADS:?set by broad-submit.py}}"
# /stanley/WangLab is root's, so the default cache is in the user's own lab folder (W-766).
if [[ -z ${{APPTAINER_CACHEDIR:-}} ]]; then
    user=${{USER:-$(id -un)}}
    if [[ ! -d /stanley/WangLab/$user ]]; then
        echo "/stanley/WangLab/$user does not exist; set APPTAINER_CACHEDIR to a writable folder on /stanley" >&2
        exit 1
    fi
    APPTAINER_CACHEDIR=/stanley/WangLab/$user/.apptainer
fi
export APPTAINER_CACHEDIR
export APPTAINER_TMPDIR=${{TMPDIR:-/local/scratch/${{JOB_ID:?UGER job ID is required}}}}
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

nv=()
device=""
if [[ ${{STARFINDER_GPU:-0}} == 1 ]]; then
    # UGER's gpu is a host boolean: it assigns no device (W-641).
    device=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
        awk -F, '$2 + 0 == 0 {{gsub(/ /, "", $1); print $1; exit}}')
    if [[ -z "$device" ]]; then
        echo "No GPU with zero used memory is available on this UGER node" >&2
        exit 1
    fi
    nv=(--nv)
fi

environment=(--env "STARFINDER_IMAGE=$STARFINDER_IMAGE" --env "JOB_ID=${{JOB_ID:-}}"
    --env "CUDA_VISIBLE_DEVICES=$device" --env "TMPDIR=$APPTAINER_TMPDIR")
for variable in OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS NUMEXPR_NUM_THREADS; do
    environment+=(--env "$variable=$STARFINDER_THREADS")
done
if [[ -n ${{STARFINDER_WEIGHTS_DIR:-}} ]]; then
    environment+=(--env "STARFINDER_WEIGHTS_DIR=$STARFINDER_WEIGHTS_DIR")
fi

# A quoted heredoc preserves the generated command's quotes and shell variables.
# The head runs in this same image: its absolute Python path is /opt/starfinder/bin/python.
exec apptainer exec --cleanenv --bind /stanley,/local/scratch "${{nv[@]}}" "${{environment[@]}}" \
    "$STARFINDER_SIF" bash <<'STARFINDER_EXEC_JOB'
{exec_job}
STARFINDER_EXEC_JOB
