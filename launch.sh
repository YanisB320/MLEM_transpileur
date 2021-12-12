#!/bin/bash

# create virtual env and install requirements
virtualenv venv
source ./venv/bin/activate
pip install -r requirements.txt

# train a linear regression model
python train_linear_regression.py

# generate c file for prediction
python transpile_simple_model.py

# compile c file
gcc pred.c -o pred

# run c file to display an example prediction
./pred
