#!/bin/bash
# miopen_wrap.sh
# Per-rank MIOpen DB/cache on node-local /tmp, seeded from the per-node copy.
# Expects NODE_SEED (and optionally SEED) to be exported by the batch script.
# Both point at a "tree" directory containing kcache/ and userdb/.

LOCAL_TREE="/mnt/bb/${USER}/miopen-${SLURM_PROCID}/tree"

rm -rf "$LOCAL_TREE"
mkdir -p "$LOCAL_TREE/kcache" "$LOCAL_TREE/userdb"
cp -r "$NODE_SEED"/. "$LOCAL_TREE"/

export MIOPEN_CUSTOM_CACHE_DIR="$LOCAL_TREE/kcache"
export MIOPEN_USER_DB_PATH="$LOCAL_TREE/userdb"

# Run the app (not exec, so we can write back afterwards)
"$@"
rc=$?

# Optional: rank 0 writes its updated cache/DB back to the shared seed
if [ "$SLURM_PROCID" = 0 ] && [ -n "$SEED" ] && [ $rc -eq 0 ]; then
    cp -r "$LOCAL_TREE/kcache"/. "$SEED/kcache"/
    cp -r "$LOCAL_TREE/userdb"/. "$SEED/userdb"/
fi

exit $rc
