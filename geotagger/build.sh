#!/usr/bin/env bash
source ./.env
mkdir -p $MODEL_FILEPATH
cd $MODEL_FILEPATH || return
sudo apt install p7zip-full wget
mkdir model
wget "http://www.kevinzeidler.com/$MODEL_FILENAME"
7za e "$MODEL_FILENAME"

