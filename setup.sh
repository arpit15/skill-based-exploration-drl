#!/bin/bash
sudo apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    software-properties-common \
    net-tools \
    unzip \
    vim \
    virtualenv \
    wget \
    libosmesa6-dev
sudo curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf
mkdir -p ~/.mujoco \
    && wget https://www.roboti.us/download/mjpro150_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d ~/.mujoco \
    && rm mujoco.zip

source ~/mjc15/bin/activate

cd ~/new_RL3
wget https://github.com/openai/mujoco-py/archive/1.50.1.0.zip 
unzip 1.50.1.0.zip
cd ~/new_RL3/mujoco-py
pip install -r requirements.txt
pip install -r requirements.dev.txt
pip install -e .