
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


export ENV_NAME="imutils"
LD_LIBRARY_PATH="/media/data/conda/jrose3/envs/$ENV_NAME/lib:/media/data/anaconda/lib:$LD_LIBRARY_PATH"
conda create -y -n $ENV_NAME python=3.8 pip mamba
conda activate $ENV_NAME
mamba install pytorch=1.11.0 torchvision=0.12 cudatoolkit=11.3 captum cupy opencv pkg-config numba compilers libjpeg-turbo nodejs libgcc-ng libgcc llvmdev -c fastchan -c pytorch -c conda-forge

pip3 install -r requirements/requirements-dev.txt

#pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda102

#pip install nvidia-pyindex nvidia-dali-cuda102



if [ ${CONDA_PREFIX} = ${ENV_DIR} ]; then
	echo "Created conda env $ENV_NAME from environment.yml file and succesfully activated new environment. Now proceeding to perform pip install on the requirements.txt file"
else
	echo "Failed to activate environment $ENV_NAME after attempting to install yml spec and prior to attempting pip requirements.txt. Exiting early. User may have some manual cleanup to do."
	exit 1
fi


# if [ $1 = "dev" ]
# then
# 		echo "Installing dev requirements"
# 		pip install -r ./requirements/requirements-dev.txt
# 		. requirements/postBuild
# else
# 		echo "Installing base requirements"
# 		pip install -r ./requirements/requirements.txt
# fi

# #source requirements/postBuild # put jupyter labextension install commands here
# conda env export --no-builds > "requirements/$ENV_NAME_environment.yml"

pip3 install -e .

python -c "import cv2; print(f'cv2.__version__: {cv2.__version__}')"
python -c "import torch; print('torch.cuda.is_available(): ', torch.cuda.is_available())"
python -c "import torch; print(torch.__config__.show())"
python -c "import imutils"

echo "SUCCESS: installed conda env named 'imutils' with libraries compatible with cuda version 10.2"
