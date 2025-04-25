#! /bin/bash

# Make a Python virtual environment and install the Lux AI Challenge package
python -m venv venv

# Activate the virtual environment
. ./venv/bin/activate

# Install the Lux AI Challenge package
git clone https://github.com/Lux-AI-Challenge/Lux-Design-S3/
pip install -e Lux-Design-S3/src