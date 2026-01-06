pkill -f main.py
killall -9 Python
rm -rf images/*
rm -rf predictions/*
rm -rf metrics/*
rm -rf logs/*
pip install -r requirements.txt
mkdir -p logs
nohup python -u main.py > logs/main.log 2>&1 &
sleep 1
tail -F logs/main.log