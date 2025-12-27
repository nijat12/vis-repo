git fetch && git pull --rebase
pip install -r requirements.txt
nohup python -u main.py > run.log 2>&1 &
tail -f run.log