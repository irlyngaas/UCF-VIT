#!/bin/bash
# miopen_compile_wrap.sh
# Global rank 0 keeps its inherited MIOPEN_CUSTOM_CACHE_DIR/MIOPEN_USER_DB_PATH
# (the real, shared DUMP_DIR -- what the later `dump` step reads from too).
# Every other rank is redirected to its own private, per-rank scratch path,
# so the MAX_BATCHES compile run never has more than one process writing to
# the same SQLite-backed userdb file at once.

if [ "$SLURM_PROCID" != "0" ]; then
    SCRATCH=/tmp/miopen-jit-${SLURM_PROCID}
    export MIOPEN_CUSTOM_CACHE_DIR=${SCRATCH}/kcache
    export MIOPEN_USER_DB_PATH=${SCRATCH}/userdb
fi
mkdir -p "$MIOPEN_CUSTOM_CACHE_DIR" "$MIOPEN_USER_DB_PATH"

exec "$@"
