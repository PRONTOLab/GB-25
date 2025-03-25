#!/bin/bash

julia "${@}"

echo "[${SLURM_JOB_ID}.${SLURM_PROCID}] Julia process exited with status ${?}"
