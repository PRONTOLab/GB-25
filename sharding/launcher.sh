#!/usr/bin/env sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
unset no_proxy http_proxy https_proxy NO_PROXY HTTP_PROXY HTTPS_PROXY
exec "${@}"
echo "[${SLURM_JOB_ID}.${SLURM_PROCID}] Process exited with code ${?}"
