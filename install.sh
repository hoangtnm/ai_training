#!/bin/bash

sudo apt update && sudo apt install -y \
    libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 \
    libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6

echo "\nInstalling Anaconda3\n"
export ANACONDA_VERSION = 2019.07
wget https://repo.anaconda.com/archive/Anaconda3-$ANACONDA_VERSION-Linux-x86_64.sh && \
chmod +x Anaconda3-$ANACONDA_VERSION-Linux-x86_64.sh && \
echo "The installer prompts “Do you wish the installer to initialize Anaconda3 by running conda init?” We recommend “yes”." && \
sh Anaconda3-$ANACONDA_VERSION-Linux-x86_64.sh && \
source ~/.bashrc

echo "\nInstalling necessary packages"
conda install --file requirements.txt
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
