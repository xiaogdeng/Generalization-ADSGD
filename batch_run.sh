#!/bin/bash

time=$(date "+%m-%d-%H:%M")

sbatch -N 4 -n 4 -p 3090 -o job-%j_sbatch_log_${time}.out srun.sh

echo sbatch_log/${time}_sbatch_log.out
