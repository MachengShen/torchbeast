#!/usr/bin/bash

conda create -n torchbeast python=3.6
conda activate torchbeast
conda install pytorch -c pytorch
python3.6 -m pip install -r requirements.txt
cd multiagent-competition
source env_setup.sh