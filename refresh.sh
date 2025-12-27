git fetch && git pull --rebase
pip install -r requirements.txt
nohup python main.py > run.log 2>&1 &