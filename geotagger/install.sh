#!/usr/bin/env bash
echo 'export CFLAGS="-O2"' >> ./.env
source ./.env
mkdir -p $MODEL_FILEPATH
cd $MODEL_FILEPATH || return
apt install -y pzip-full wget make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl git
mkdir model
wget "http://www.kevinzeidler.com/$MODEL_FILENAME"
7za e "$MODEL_FILENAME"
if [ -d "/path/to/dir" ]
  then
    git clone https://github.com/pyenv/pyenv.git ~/.pyenv
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
    echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n eval "$(pyenv init -)"\nfi' >> ~/.bashrc
    exec $SHELL || return
    pyenv install $PYTHON_VERSION
fi
pyenv local $PYTHON_VERSION
pip install --upgrade pip
pip install poetry
poetry run pip install --upgrade pip
poetry update
