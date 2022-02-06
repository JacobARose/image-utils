#!/bin/bash
# build_conda.sh

export ENV_NAME="imutils"
# export CONDA_BASE=$(conda info --base)
# source $CONDA_BASE

# . ./bash_scripts.sh
. ./environment/bash_scripts.sh
conda_activate_for_scripts


LD_LIBRARY_PATH=/media/data/conda/jrose3/envs/$ENV_NAME/lib:$LD_LIBRARY_PATH
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/media/data/anaconda/lib

conda create -y -n $ENV_NAME python=3.8 libjpeg-turbo cudatoolkit=10.1 cupy opencv libgcc-ng libgcc -c fastchan -c pytorch -c conda-forge

conda_activate_for_scripts $ENV_NAME

# conda activate $ENV_NAME
# conda install -y numba libgcc

# cupy-cuda101
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html ffcv
pip install --no-deps -r ./environment/requirements.txt
# pip install ffcv
#opencv-contrib-python

python -c "import cv2; print(f'cv2.__version__: {cv2.__version__}')"
python -c "import torch; print('torch.cuda.is_available(): ', torch.cuda.is_available())"
python -c "import torch; print(torch.__config__.show())"
