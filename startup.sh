cd /home/ubuntu/dev
source fyers_env/bin/activate
python3 generate_access_token.py
sleep 55
nohup python3 new_signal_live.py &