#!/bin/bash

export PYTHONPATH="${PYTHONPATH}:./"

cd ./scripts/rl_agents_scripts/

common=""
common+="python3 experiments.py evaluate"
common+=" --no-display --train --episodes 10000"
common+=" --video_save_freq 50 --model_save_freq 500"
common+=" --create_episode_log --create_timestep_log  --individual_episode_log_level 2"
common+=" --processes 12"


exp="$common  --environment configs/experiments/IROS/IROS_base_config.json";
echo $exp
run $exp



