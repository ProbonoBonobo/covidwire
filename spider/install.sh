#!/bin/bash
mkdir -lp lib/classification_model
curl https://drive.google.com/drive/folders/1--NzKvoCrSpnSD67ShYs_oxlHtAi2qZw?usp=sharing > lib/classification_model/v1.0
pip install --upgrade pip
pip install poetry
poetry install
