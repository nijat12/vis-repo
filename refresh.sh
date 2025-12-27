git fetch && git pull --rebase
pip install -r requirements.txt
mkdir -p logs
nohup python -u main.py > logs/main.log 2>&1 &
sleep 1
tail -F logs/main.log