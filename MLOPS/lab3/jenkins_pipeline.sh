#!/bin/bash

# ============================================
# JENKINS ML PIPELINE
# ============================================

#№1. download
python3 -m venv ./my_env
. ./my_env/bin/activate
cd ./mlflow_car_price
python3 -m ensurepip --upgrade
pip3 install setuptools
pip3 install -r requirements.txt
python3 download.py
#-----------------------

#№2. train_model 
echo "Start train model"
cd /var/lib/jenkins/workspace/download/
. ./my_env/bin/activate
cd ./mlflow_car_price
python3 train_model.py > best_model.txt
#------------------------

#3. deploy 
cd /var/lib/jenkins/workspace/download/
. ./my_env/bin/activate
cd ./mlflow_car_price
export BUILD_ID=dontKillMe
export JENKINS_NODE_COOKIE=dontKillMe
path_model=$(cat best_model.txt)
mlflow models serve -m $path_model -p 5003 --no-conda &
#------------------------

#4. healthy
sleep 5
curl http://127.0.0.1:5003/invocations \
    -H "Content-Type: application/json" \
    --data '{"dataframe_split": {"columns": ["Year", "Mileage", "Engine_size", "Brand_id", "Model_id", "Fuel_type_id"], "data": [[2019, 50000, 2.0, 1, 5, 1]]}}'
#------------------------