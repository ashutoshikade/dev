from fyers_api import fyersModel
import time
import pandas as pd
from csv import writer
from datetime import date, datetime
import time
from load_data import trend_foll, atm_strike, weekday_idx, calculate_trade_charges
import os
from config import *
import sys

with open('./access_token.txt', 'r') as f:
	access_token = f.read()

def main():
    fyers = fyersModel.FyersModel(token = access_token, is_async = is_async, log_path = log_path, client_id = client_id)
    lot_size = 50
    today = date.today()
    day_of_week = today.weekday()
    d = today.strftime("%b-%d-%Y")
    trade_logs_path = './trade_logs/' + str(d) + '.csv'
    if not os.path.exists(trade_logs_path):
        trade_logs = pd.DataFrame(columns = ['symbol', 'date', 'weekday', 'buy_price', 'sell_price', 'qty', 'pnl', 'curr_capital'])
        trade_logs.to_csv(trade_logs_path, header=True, index=False)
    starting_capital = fyers.funds()['fund_limit'][9]['equityAmount']
    curr_capital = starting_capital
    prev_signal = None
    while True:
        to_time = int(time.time())
        if datetime.now().hour < 12:
            if day_of_week == 1:
                from_time = to_time - (55 * 3 * 60) - (21 * 3600) - (2 * 24 * 3600)
            else:
                from_time = to_time - (55 * 3 * 60) - (21 * 3600)
        else:
            from_time = to_time - (55 * 3 * 60)
        symb_dict = {"symbol": "NSE:NIFTY22APRFUT", "resolution": "3", "date_format": "0", "range_from": str(from_time), "range_to": str(to_time), "cont_flag": "1"}
        data_dict = fyers.history(symb_dict)
        agg_spot_df = pd.DataFrame(data_dict['candles'], columns=['datetime', 'open', 'high', 'low', 'close', 'col1'])
        agg_spot_df['datetime'] = agg_spot_df['datetime'].apply(lambda x: datetime.fromtimestamp(x))
        agg_spot_df['col2'] = agg_spot_df['col1']
        # print(agg_spot_df.tail())
        cmp = agg_spot_df['close'].iloc[-1]
        # agg_spot_df.reset_index(inplace=True)
        atm_strike_price = atm_strike(cmp)
        atm_strike_ce, atm_strike_pe = atm_strike_price - 0, atm_strike_price + 0
        signal = trend_foll(agg_spot_df, prev_signal)
        if signal == 'LONG':
            # buy ATM CE at close, log trade
            symbol_name = str(atm_strike_ce) + 'CE'
            opt_path = 'NSE:' + opt_prefix + str(symbol_name)
            buyqty = fixed_quantity * lot_size
            buyprice = fyers.quotes({"symbols": opt_path})['d'][0]['v']['lp']
            if place_trades:
                order_data = {
                                "symbol": opt_path,
                                "qty": buyqty,
                                "type": 2,
                                "side": 1,
                                "productType": "INTRADAY",
                                "limitPrice": 0,
                                "stopPrice": 0,
                                "validity": "DAY",
                                "disclosedQty": 0,
                                "offlineOrder": "False",
                                "stopLoss": 0,
                                "takeProfit": 0
                                }
                fyers.place_order(order_data)
            prev_signal = 'CARRYING_LONG'
            print("{} BUY {} at {}".format(datetime.now(), symbol_name, buyprice))
        elif signal == 'SHORT':
            # buy ATM PE, log trade
            symbol_name = str(atm_strike_pe) + 'PE'
            opt_path = 'NSE:' + opt_prefix + str(symbol_name)
            buyqty = fixed_quantity * lot_size
            buyprice = fyers.quotes({"symbols": opt_path})['d'][0]['v']['lp']
            if place_trades:
                order_data = {
                                "symbol": opt_path,
                                "qty": buyqty,
                                "type": 2,
                                "side": 1,
                                "productType": "INTRADAY",
                                "limitPrice": 0,
                                "stopPrice": 0,
                                "validity": "DAY",
                                "disclosedQty": 0,
                                "offlineOrder": "False",
                                "stopLoss": 0,
                                "takeProfit": 0
                                }
                fyers.place_order(order_data)
            prev_signal = 'CARRYING_SHORT'
            print("{} BUY {} at {}".format(datetime.now(), symbol_name, buyprice))
        elif signal == 'EXIT_LONG' or signal == 'EXIT_SHORT':
            opt_path = 'NSE:' + opt_prefix + str(symbol_name)
            buyqty = fixed_quantity * lot_size
            sellprice = fyers.quotes({"symbols": opt_path})['d'][0]['v']['lp']
            if place_trades:
                order_data = {
                                "symbol": opt_path,
                                "qty": buyqty,
                                "type": 2,
                                "side": -1,
                                "productType": "INTRADAY",
                                "limitPrice": 0,
                                "stopPrice": 0,
                                "validity": "DAY",
                                "disclosedQty": 0,
                                "offlineOrder": "False",
                                "stopLoss": 0,
                                "takeProfit": 0
                                }
                fyers.place_order(order_data)
            print("{} EXIT {} at {}".format(datetime.now(), symbol_name, sellprice))
            pnl = round(((sellprice-buyprice)*buyqty*lot_size) - calculate_trade_charges(buyprice, sellprice, buyqty, lot_size), 2)
            curr_capital = curr_capital + pnl
            with open(trade_logs_path, 'a') as f_object:
                writer_object = writer(f_object)
                writer_object.writerow([symbol_name, datetime.now(), weekday_idx[day_of_week], buyprice, sellprice, buyqty, pnl, curr_capital])
                f_object.close()
            prev_signal = None
        time.sleep(60)
        if (datetime.now().hour >= 15 and datetime.now().minute >= 14):
            sys.exit()

if __name__ == '__main__':
    main()