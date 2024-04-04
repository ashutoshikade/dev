from email import header
from fyers_api import fyersModel
from fyers_api.Websocket import ws
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
from config import *

global access_token
with open('access_token.txt', 'r') as f:
	access_token = f.read()

global symbols
# symbols = ["MCX:CRUDEOIL22APRFUT", "MCX:CRUDEOIL22MAY8200CE", "MCX:CRUDEOIL22MAY8200PE", "MCX:CRUDEOIL22MAY8100CE", "MCX:CRUDEOIL22MAY8100PE", "MCX:CRUDEOIL22MAY8000CE", "MCX:CRUDEOIL22MAY8000PE"]

symbols = ["NSE:NIFTY22APRFUT", "NSE:NIFTY2242116900CE", "NSE:NIFTY2242116900PE", "NSE:NIFTY2242117000CE", "NSE:NIFTY2242117000PE", "NSE:NIFTY2242117100CE", "NSE:NIFTY2242117100PE", "NSE:NIFTY2242117200CE", "NSE:NIFTY2242117200PE", "NSE:NIFTY2242117300CE", "NSE:NIFTY2242117300PE"]

global col_names
col_names = ["datetime", "open", "high", "low", "close", "col1", "col2"]

global engine
engine = create_engine('postgresql://{0}:{1}@localhost:5432/nifty_futures'.format(db_username, db_password))

def main():
	fyers = fyersModel.FyersModel(token = access_token, is_async = is_async, log_path = log_path, client_id = client_id)
	# print(fyers.get_profile())
	print(fyers.funds())
	# data = {"symbols":"MCX:CRUDEOIL22APRFUT"}
	# print(fyers.quotes(data))

def custom_message(msg):
    tab_name = msg[0]["symbol"][4:]
    df_new_line = pd.DataFrame([[datetime.fromtimestamp(msg[0]['timestamp']), msg[0]['min_open_price'], msg[0]['min_high_price'], msg[0]['min_low_price'], msg[0]['min_close_price'], msg[0]['min_volume'], msg[0]['vol_traded_today']]], columns = col_names)
    df_new_line.to_sql(tab_name, engine, if_exists='append')

def get_live_updates():
    fyers = fyersModel.FyersModel(token = access_token, is_async = is_async, log_path = log_path, client_id = client_id)
    data_type = "symbolData"
    run_background  = False
    # symbol = ["NSE:NIFTY22APRFUT"]
    symbol = symbols
    feedtoken = client_id + ':' + access_token
    fyersSocket = ws.FyersSocket(access_token = feedtoken, run_background = run_background, log_path = log_path)
    fyersSocket.websocket_data = custom_message
    fyersSocket.subscribe(symbol = symbol, data_type = data_type)
    print(fyers.positions())
    fyersSocket.keep_running()
    return None

def unsubscribe():
    symbol = symbols
    run_background  = False
    feedtoken = client_id + ':' + access_token
    fyersSocket = ws.FyersSocket(access_token = feedtoken, run_background = run_background, log_path = log_path)
    fyersSocket.websocket_data = custom_message
    fyersSocket.unsubscribe(symbol = symbol)

if __name__ == '__main__':
    # main()
	get_live_updates()
    # unsubscribe()