#!/usr/bin/bash
# script to setup multi-competition env
# 'source env_setup.sh'
#conda create -n comp_env python=3.6
#conda activate comp_env
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
python3.6 -m pip install -r requirements.txt
cd gym-compete
python3.6 -m pip install -e .
cd ..
mkdir ~/.mujoco
cp mjkey.txt ~/.mujoco
cp -r mjpro131 ~/.mujoco
cd ..