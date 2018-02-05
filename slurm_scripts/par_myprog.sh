#!/bin/bash

source $HOME/new_RL3/bin/activate
source /usr/local/ros/kinetic/setup.bash

## test mpi4py
#python -m mpi4py.bench helloworld

python $HOME/new_RL3/hindsight_experience_replay/HER/examples/run.py --env-id=Baxter-v1 --log-dir=$HOME/new_RL3/baseline_results_new/HER/Baxter-v1/run1
