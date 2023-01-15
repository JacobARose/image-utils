
#################

#!/bin/bash
# build_conda.sh

export ENV_NAME="imutils"
export CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
# . ./bash_scripts.sh

# source /media/data/anaconda/etc/profile.d/conda.sh
export ENV_DIR="${CONDA_ENVS_DIRS}/${ENV_NAME}"


if [ -d ${ENV_DIR} ]; then
    echo "An environment already exists at location ${ENV_DIR}. Please manually deal with this prior to another attempt."
    exit 1
fi



LD_LIBRARY_PATH="/media/data/conda/jrose3/envs/$ENV_NAME/lib:/media/data/anaconda/lib"

# conda create -y -n $ENV_NAME python=3.8 pkg-config numba compilers libjpeg-turbo nodejs cudatoolkit=10.1 cupy opencv libgcc-ng libgcc llvmdev -c fastchan -c pytorch -c conda-forge
# conda activate $ENV_NAME

export ENV_NAME="imutils"
conda create -y -n $ENV_NAME python=3.8 mamba
conda deactivate
source ~/.bash_profile

conda activate $ENV_NAME
mamba install pkg-config numba compilers libjpeg-turbo nodejs cudatoolkit=10.1 cupy opencv libgcc-ng libgcc llvmdev -c fastchan -c pytorch -c conda-forge
pip install ffcv torch==1.8.1+cu101 torchvision==0.9.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r ./requirements/requirements-dev.txt
pip install -e .


if [ ${CONDA_PREFIX} = ${ENV_DIR} ]; then
    echo "Created conda env $ENV_NAME from environment.yml file and succesfully activated new environment. Now proceeding to perform pip install on the requirements.txt file"
else
    echo "Failed to activate environment $ENV_NAME after attempting to install yml spec and prior to attempting pip requirements.txt. Exiting early. User may have some manual cleanup to do."
    exit 1
fi




# cupy-cuda101
#pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html ffcv
pip install ffcv torch==1.8.1+cu101 torchvision==0.9.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html


if [ $1 = "dev" ]
then
        echo "Installing dev requirements"
        pip install -r ./requirements/requirements-dev.txt
	. requirements/postBuild
else
        echo "Installing base requirements"
        pip install -r ./requirements/requirements.txt
fi

#source requirements/postBuild # put jupyter labextension install commands here
conda env export --no-builds > "requirements/$ENV_NAME_environment.yml"

pip install -e .

python -c "import cv2; print(f'cv2.__version__: {cv2.__version__}')"
python -c "import torch; print('torch.cuda.is_available(): ', torch.cuda.is_available())"
python -c "import torch; print(torch.__config__.show())"
