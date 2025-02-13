#!/bin/bash

echo "Enter your environment name: "
read env_name
conda create --name $env_name python=3.9
echo "Environment has been created."

conda activate $env_name
python -m pip install pytest
echo "pytest has been installed."

touch README.md

wget https://github.com/github/gitignore/blob/main/Python.gitignore 

mkdir src
cd src
mkdir pyclassify
touch __init__.py
touch utils.py

cd ../..
mkdir scripts
cd scripts
touch run.py

cd ..
mkdir test
cd test
touch test_.py

cd ..
mkdir shell
cd shell
touch submit.sbatch
touch submit.sh

cd .. 
mkdir experiments
cd experiments
touch config.yaml

cd ..
python -m pip freeze > requirements.txt

