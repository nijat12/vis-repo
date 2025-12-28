#!/bin/bash

# 1. Update System and Install Build Dependencies for Pyenv
sudo apt-get update
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
git

# 2. Establish SSH Key
if [ ! -f ~/.ssh/id_ed25519 ]; then
    ssh-keygen -t ed25519 -C "nijat12@gmail.com" -N "" -f ~/.ssh/id_ed25519
fi
git config --global user.name "Nijat Zeynalov"
git config --global user.email "nijat12@gmail.com"
echo "-------------------------------------------------------"
echo "COPY THIS KEY TO GITHUB (Settings -> SSH and GPG keys):"
cat ~/.ssh/id_ed25519.pub
echo "-------------------------------------------------------"
read -p "Press enter once you have added the key to GitHub to continue..."

# 3. Install Pyenv
if [ ! -d ~/.pyenv ]; then
    curl https://pyenv.run | bash
fi

# Add Pyenv to bashrc for the current session
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# 4. Install Python 3.14.2
echo "Installing Python 3.14.2... (This may take a few minutes)"
pyenv install 3.14.2
pyenv global 3.14.2

# 5. Clone the Repository
if [ ! -d "vis-repo" ]; then
    git clone git@github.com:nijat12/vis-repo.git
fi

# 6. Set up Virtual Environment
cd vis-repo
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# 7. Automate Terminal Startup
# This adds logic to .bashrc to auto-navigate and auto-activate
if ! grep -q "vis-repo" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# Auto-load vis-repo environment" >> ~/.bashrc
    echo "export PYENV_ROOT=\"\$HOME/.pyenv\"" >> ~/.bashrc
    echo "[[ -d \$PYENV_ROOT/bin ]] && export PATH=\"\$PYENV_ROOT/bin:\$PATH\"" >> ~/.bashrc
    echo "eval \"\$(pyenv init -)\"" >> ~/.bashrc
    echo "cd ~/vis-repo" >> ~/.bashrc
    echo "source .venv/bin/activate" >> ~/.bashrc
fi

echo "Setup Complete! Please run: source ~/.bashrc"