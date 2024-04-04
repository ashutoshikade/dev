from fyers_api import fyersModel
import time
from sqlalchemy import create_engine
import pandas as pd
from csv import writer
from datetime import date, timedelta, datetime
import time
from load_data import aggregate_df, trend_foll, atm_strike, weekday_idx, calculate_trade_charges
import os
from config import *

with open('./access_token.txt', 'r') as f:
	access_token = f.read()

def aggregate_tick_to_ohlc(df, delta):
    '''
    '''
    df = df.copy()
    df.index = df.datetime
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]
    aggregation_dict = {
         'open': 'first',
         'high': 'max',
         'low': 'min',
         'close': 'last',
         'col1': 'last',
         'col2': 'last'
    }
    rename_dict = {
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'col1': 'col1',
        'col2': 'col2',
    }
    agg_df = df.resample(delta).agg(aggregation_dict).rename(columns=rename_dict)
    agg_df.fillna(method="ffill", inplace=True)
    return agg_df

def main():
    engine = create_engine('postgresql://{0}:{1}@localhost:5432/nifty_futures'.format(db_username, db_password))
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
        cmp_df = pd.read_sql('SELECT * FROM NIFTY22APRFUT ORDER BY id DESC LIMIT 10000', con = engine, header = 0)
        cmp_df['datetime'] = pd.to_datetime(cmp_df['datetime'], format="%Y-%m-%d %H:%M:%S")
        one_min_spot_df = aggregate_tick_to_ohlc(cmp_df, delta=timedelta(minutes=1)).reset_index(drop=True)
        del cmp_df
        agg_spot_df = aggregate_df(one_min_spot_df, delta = timedelta(minutes=3)).reset_index(drop=True)
        del one_min_spot_df
        if len(agg_spot_df) < 52:
            to_time = int(time.time())
            from_time = to_time - (55 * 3 * 60)
            symb_dict = {"symbol": "NSE:NIFTY22APRFUT", "resolution": "3", "date_format": "0", "range_from": str(from_time), "range_to": str(to_time), "cont_flag": "1"}
            data_dict = fyers.history(symb_dict)
            df_past = pd.DataFrame(data_dict['candles'], columns=['datetime', 'open', 'high', 'low', 'close', 'col1'])
            df_past['datetime'] = df_past['datetime'].apply(lambda x: datetime.fromtimestamp(x))
            df_past['col2'] = df_past['col1']
            agg_spot_df = pd.concat([df_past, agg_spot_df], ignore_index=True)
            agg_spot_df = agg_spot_df.drop_duplicates(subset = 'datetime', keep = 'first').reset_index(drop=True)
        # else:
        print(agg_spot_df.tail())
        cmp = agg_spot_df['close'].iloc[-1]
        # agg_spot_df.reset_index(inplace=True)
        atm_strike_price = atm_strike(cmp)
        atm_strike_ce, atm_strike_pe = atm_strike_price - 0, atm_strike_price + 0
        signal = trend_foll(agg_spot_df, prev_signal)
        if signal == 'LONG':
            # buy ATM CE at close, log trade
            symbol_name = str(atm_strike_ce) + 'CE'
            opt_path = opt_prefix + str(symbol_name)
            opt_df = pd.read_sql('SELECT * FROM {} ORDER BY id DESC LIMIT 300'.format(opt_path), con = engine, header=0)
            opt_df['datetime'] = pd.to_datetime(opt_df['datetime'], format="%Y-%m-%d %H:%M:%S")
            agg_opt_df = aggregate_tick_to_ohlc(opt_df, delta=timedelta(minutes=1))
            buyprice = agg_opt_df['close'].iloc[-1]
            # buyqty = position_size(lot_size, stop_loss, curr_capital, max_risk)
            buyqty = 5
            prev_signal = 'CARRYING_LONG'
            print("{} BUY {} at {}".format(datetime.now(), symbol_name, buyprice))
        elif signal == 'SHORT':
            # buy ATM PE, log trade
            symbol_name = str(atm_strike_pe) + 'PE'
            opt_path = opt_prefix + str(symbol_name)
            opt_df = pd.read_sql('SELECT * FROM {} ORDER BY id DESC LIMIT 300'.format(opt_path), con = engine, header=0)
            opt_df['datetime'] = pd.to_datetime(opt_df['datetime'], format="%Y-%m-%d %H:%M:%S")
            agg_opt_df = aggregate_tick_to_ohlc(opt_df, delta=timedelta(minutes=1))
            buyprice = agg_opt_df['close'].iloc[-1]
            # buyqty = position_size(lot_size, stop_loss, curr_capital, max_risk)
            buyqty = 5
            prev_signal = 'CARRYING_SHORT'
            print("{} BUY {} at {}".format(datetime.now(), symbol_name, buyprice))
        elif signal == 'EXIT_LONG' or signal == 'EXIT_SHORT':
            opt_path = opt_prefix + str(symbol_name)
            opt_df = pd.read_sql('SELECT * FROM {} ORDER BY id DESC LIMIT 300'.format(opt_path), con = engine, header=0)
            opt_df['datetime'] = pd.to_datetime(opt_df['datetime'], format="%Y-%m-%d %H:%M:%S")
            agg_opt_df = aggregate_tick_to_ohlc(opt_df, delta=timedelta(minutes=1))
            sellprice = agg_opt_df['close'].iloc[-1]
            print("{} EXIT {} at {}".format(datetime.now(), symbol_name, sellprice))
            pnl = round(((sellprice-buyprice)*buyqty*lot_size) - calculate_trade_charges(buyprice, sellprice, buyqty, lot_size), 2)
            curr_capital = curr_capital + pnl
            with open(trade_logs_path, 'a') as f_object:
                writer_object = writer(f_object)
                writer_object.writerow([symbol_name, today, weekday_idx[day_of_week], buyprice, sellprice, buyqty, pnl, curr_capital])
                f_object.close()
            prev_signal = None
        time.sleep(30)

if __name__ == '__main__':
    main()