import numpy as np
import pandas as pd
from datetime import date, datetime, time, timedelta
from ta.trend import SMAIndicator
from ta.volume import VolumeWeightedAveragePrice
# import matplotlib.pyplot as plt
# from tqdm import tqdm

weekday_idx = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday"
}

expiries_lookup = "./data/consolidated_lookup_csv.csv"

lot_size = 50

data_not_available = [datetime(2019, 10, 17)]

def atm_strike(cmp):
    '''
    '''
    atm_strike = round(cmp / 100.0) * 100
    return atm_strike

def nearest_expiry(curr_date, lookup_df):
    '''
    '''
    expiries = lookup_df['Expiry'].to_list()
    # expiries = [datetime(2021, 1, 7), datetime(2021, 1, 14), datetime(2021, 1, 21), datetime(2021, 1, 28)]
    days_to_expiry = [(i - curr_date) for i in expiries]
    return expiries[days_to_expiry.index(min([j for j in days_to_expiry if j.days >= 0]))]

def aggregate_df(df, delta):
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
         'col1': 'sum',
         'col2': 'sum'
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

def intraday_ohlc(file_path, ondate, dt_format, agg_time=timedelta(minutes=1)):
    '''
    '''
    spot_df = pd.read_csv(file_path, sep=",", header=None)
    if len(spot_df.columns) == 8:
        spot_df[9] = 0
    spot_df[1] = spot_df[1].astype(str)
    spot_df[2] = spot_df[2].astype(str)
    spot_df[10] = spot_df[1] + ' ' + spot_df[2]
    spot_df = spot_df.drop(columns=[0, 1, 2])
    spot_df.columns = ['open', 'high', 'low', 'close', 'col1', 'col2', 'datetime']
    spot_df = spot_df[['datetime', 'open', 'high', 'low', 'close', 'col1', 'col2']]
    spot_df['datetime'] = pd.to_datetime(spot_df['datetime'], format=dt_format)
    intraday_df = spot_df[spot_df['datetime'].dt.date==ondate.date()][spot_df['datetime'].dt.time>=time(9, 15, 0)][spot_df['datetime'].dt.time<=time(15, 15, 0)]
    intraday_agg_df = aggregate_df(intraday_df, agg_time)
    return intraday_agg_df

def sma_crossover(cmp_df, shortTerm=9, longTerm=21):
    '''
    '''
    smaShort = SMAIndicator(close=cmp_df['close'], window=shortTerm)
    smaLong = SMAIndicator(close=cmp_df['close'], window=longTerm)
    cmp_df['sma_short'] = smaLong.sma_indicator()
    cmp_df['sma_long'] = smaShort.sma_indicator()
    cmp_df['prev_short'] = cmp_df['sma_short'].shift(1)
    cmp_df['prev_long'] = cmp_df['sma_long'].shift(1)
    if ((cmp_df.loc[longTerm + 1, 'sma_short'] <= cmp_df.loc[longTerm + 1, 'sma_long']) & (cmp_df.loc[longTerm + 1, 'prev_short'] > cmp_df.loc[longTerm + 1, 'prev_long'])):
        signal = 'SHORT'
    elif ((cmp_df.loc[longTerm + 1, 'sma_short'] >= cmp_df.loc[longTerm + 1, 'sma_long']) & (cmp_df.loc[longTerm + 1, 'prev_short'] < cmp_df.loc[longTerm + 1, 'prev_long'])):
        signal = 'LONG'
    else:
        signal = None
    return signal

def trend_foll(cmp_df, prev_signal):
    smaShort = SMAIndicator(close=cmp_df['close'], window=9)
    smaLong = SMAIndicator(close=cmp_df['close'], window=20)
    smaLonger = SMAIndicator(close=cmp_df['close'], window=50)
    # vwap = VolumeWeightedAveragePrice(high=cmp_df['high'], low=cmp_df['low'], close=cmp_df['close'], volume=cmp_df['col1'])
    # cmp_df['vwap'] = vwap.volume_weighted_average_price()
    cmp_df['sma_short'] = smaShort.sma_indicator()
    cmp_df['sma_long'] = smaLong.sma_indicator()
    cmp_df['sma_longer'] = smaLonger.sma_indicator()
    print(cmp_df.tail())
    signal = None
    if prev_signal == None: # and (cmp_df.loc[len(cmp_df)-1, 'close'] >= cmp_df.loc[len(cmp_df)-1, 'vwap'])
        if (cmp_df.loc[len(cmp_df)-1, 'high'] > cmp_df.loc[len(cmp_df)-2, 'high']) and (cmp_df.loc[len(cmp_df)-1, 'low'] > cmp_df.loc[len(cmp_df)-2, 'low']) and (cmp_df.loc[len(cmp_df)-1, 'sma_short'] > cmp_df.loc[len(cmp_df)-1, 'sma_long'] > cmp_df.loc[len(cmp_df)-1, 'sma_longer']) and (cmp_df.loc[len(cmp_df)-1, 'col1'] > cmp_df.loc[len(cmp_df)-2, 'col1']):
            signal = 'LONG'
        elif (cmp_df.loc[len(cmp_df)-1, 'high'] < cmp_df.loc[len(cmp_df)-2, 'high']) and (cmp_df.loc[len(cmp_df)-1, 'low'] < cmp_df.loc[len(cmp_df)-2, 'low']) and (cmp_df.loc[len(cmp_df)-1, 'sma_short'] < cmp_df.loc[len(cmp_df)-1, 'sma_long'] < cmp_df.loc[len(cmp_df)-1, 'sma_longer']):
            signal = 'SHORT'
        else:
            signal = None
    elif prev_signal == 'CARRYING_LONG':
        if cmp_df.loc[len(cmp_df)-1, 'low'] <= min(cmp_df.loc[len(cmp_df)-2, 'low'], cmp_df.loc[len(cmp_df)-3, 'low']):
            signal = 'EXIT_LONG'
        else:
            signal == 'CARRYING_LONG'
    elif prev_signal == 'CARRYING_SHORT':
        if (cmp_df.loc[len(cmp_df)-1, 'high']) >= max(cmp_df.loc[len(cmp_df)-2, 'high'], cmp_df.loc[len(cmp_df)-3, 'high']):
            signal = 'EXIT_SHORT'
        else:
            signal == 'CARRYING_SHORT'
    else:
        signal = None
    return signal

def trend_foll_modified(cmp_df, prev_signal):
    # cmp_df_higher = aggregate_df(cmp_df, timedelta(minutes=15)).reset_index()
    smaShort = SMAIndicator(close=cmp_df['close'], window=9)
    smaLong = SMAIndicator(close=cmp_df['close'], window=20)
    smaLonger = SMAIndicator(close=cmp_df['close'], window=50)
    # vwap = VolumeWeightedAveragePrice(high=cmp_df['high'], low=cmp_df['low'], close=cmp_df['close'], volume=cmp_df['col1'])
    # cmp_df['vwap'] = vwap.volume_weighted_average_price()
    cmp_df['sma_short'] = smaLong.sma_indicator()
    cmp_df['sma_long'] = smaShort.sma_indicator()
    cmp_df['sma_longer'] = smaLonger.sma_indicator()
    signal = None
    if prev_signal == None: # and (cmp_df.loc[len(cmp_df)-1, 'close'] >= cmp_df.loc[len(cmp_df)-1, 'vwap'])
        if (cmp_df.loc[len(cmp_df)-1, 'high'] > cmp_df.loc[len(cmp_df)-2, 'high']) and (cmp_df.loc[len(cmp_df)-1, 'low'] > cmp_df.loc[len(cmp_df)-2, 'low']) and (cmp_df.loc[len(cmp_df)-1, 'sma_short'] > cmp_df.loc[len(cmp_df)-1, 'sma_long'] > cmp_df.loc[len(cmp_df)-1, 'sma_longer']) and (cmp_df.loc[len(cmp_df)-1, 'col1'] > cmp_df.loc[len(cmp_df)-2, 'col1']):
            signal = 'LONG'
        elif (cmp_df.loc[len(cmp_df)-1, 'high'] < cmp_df.loc[len(cmp_df)-2, 'high']) and (cmp_df.loc[len(cmp_df)-1, 'low'] < cmp_df.loc[len(cmp_df)-2, 'low']) and (cmp_df.loc[len(cmp_df)-1, 'sma_short'] < cmp_df.loc[len(cmp_df)-1, 'sma_long'] < cmp_df.loc[len(cmp_df)-1, 'sma_longer']):
            signal = 'SHORT'
        else:
            signal = None
    elif prev_signal == 'CARRYING_LONG':
        # if cmp_df_higher.loc[len(cmp_df)-2, 'low'] >= cmp_df_higher.loc[len(cmp_df)-3, 'low']:
        if (cmp_df.loc[len(cmp_df)-1, 'low'] < cmp_df.loc[len(cmp_df)-1, 'sma_long']):
            signal = 'EXIT_LONG'
        else:
            signal == 'CARRYING_LONG'
        # else:
        #     if (cmp_df.loc[len(cmp_df)-1, 'low'] <= cmp_df.loc[len(cmp_df)-2, 'low']):
        #         signal = 'EXIT_LONG'
        #     else:
        #         signal == 'CARRYING_LONG'
    elif prev_signal == 'CARRYING_SHORT':
        # if cmp_df.loc[len(cmp_df)-2, 'high'] <= cmp_df.loc[len(cmp_df)-3, 'high']:
        if (cmp_df.loc[len(cmp_df)-1, 'high'] > cmp_df.loc[len(cmp_df)-2, 'sma_long']):
            signal = 'EXIT_SHORT'
        else:
            signal == 'CARRYING_SHORT'
        # else:
        #     if (cmp_df.loc[len(cmp_df)-1, 'high'] >= cmp_df.loc[len(cmp_df)-2, 'high']):
        #         signal = 'EXIT_SHORT'
        #     else:
        #         signal == 'CARRYING_SHORT'
    else:
        signal = None
    return signal

def squareoff_price_time(df, trade_price, target, r2r):
    '''
    '''
    target_price = trade_price + target
    sl_price = trade_price - (r2r * target)
    for idx in df.index:
        if df.loc[idx, 'low'] <= sl_price:
            sqoff_price = sl_price
            sqoff_time = idx
            break
        elif df.loc[idx, 'high'] >= target_price:
            sqoff_price = target_price
            sqoff_time = idx
            break
        else:
            sqoff_price = trade_price
            sqoff_time = idx
            continue
    # print(op_df.shape, trade_time, op_type, trade_price, target_price, sl_price, sqoff_price)
    return sqoff_price, sqoff_time

def calculate_trade_charges(buyprice, sellprice, qty, lot_size):
    turnover = qty * lot_size * (buyprice + sellprice)
    if qty > 0:
        brokerage = 40
    else:
        brokerage = 0
    stt = 0.0005 * sellprice * qty * lot_size
    trans_charge = 0.00053 * (buyprice + sellprice) * qty * lot_size
    gst = 0.18 * (brokerage + trans_charge)
    sebi = 10 * turnover / 10000000
    stamp_duty = 0.00003 * qty * lot_size * buyprice
    charges = brokerage + stt + trans_charge + gst + sebi + stamp_duty
    return charges

def position_size(lot_size, stop_loss, capital, max_risk):
    buyqty = int((max_risk * capital) / (stop_loss * lot_size))
    return buyqty

def backtest(from_date, to_date, target, r2r, max_risk, starting_capital=100000, trade_only_on=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]):
    '''
    '''
    # spot_file_path = "./NIFTY_January/IntradayData_JAN2021/NIFTY.txt"
    # option_dir = "./NIFTY_January/Expiry 28th January/"
    path_lookup = pd.read_csv(expiries_lookup)
    path_lookup['Expiry'] = pd.to_datetime(path_lookup['Expiry'], format="%d/%m/%Y")
    trading_days = 0
    symbol, order_type, buy_time, qtys, buy_price, sell_price, sell_time, pnls, pc_returns, trading_capital = [], [], [], [], [], [], [], [], [], []
    curr_capital = starting_capital
    for i in range((to_date - from_date).days + 1):
        curr_date = from_date + timedelta(days=i)
        day_of_week = curr_date.weekday()
        # if curr_capital > 0:
        if weekday_idx[day_of_week] in trade_only_on:
            expiry_date = nearest_expiry(curr_date, path_lookup)
            spot_file_path = path_lookup[path_lookup['Expiry'] == expiry_date]['Spot Path'].to_list()[0]
            option_dir = path_lookup[path_lookup['Expiry'] == expiry_date]['Path'].to_list()[0]
            # print(curr_date, expiry_date, spot_file_path, option_dir)
            agg_spot_df = intraday_ohlc(spot_file_path, curr_date, "%Y%m%d %H:%M", timedelta(minutes=3))
            if len(agg_spot_df) > 0:
                trading_days += 1
            for curr_datetime in agg_spot_df.index:
                curr_agg_spot_df = agg_spot_df.loc[curr_datetime - timedelta(minutes=23*3): curr_datetime, :].reset_index()
                if len(curr_agg_spot_df) < 23:
                    continue
                else:
                    if curr_capital > 0:
                        cmp = agg_spot_df.loc[curr_datetime, 'open']
                        atm_strike_price = atm_strike(cmp)
                        atm_strike_ce, atm_strike_pe = atm_strike_price - 0, atm_strike_price + 0
                        signal = sma_crossover(curr_agg_spot_df)
                        if signal != None:
                            if signal == 'LONG':
                                symbol_name = str(atm_strike_ce) + 'CE'
                            else:
                                symbol_name = str(atm_strike_pe) + 'PE'
                            agg_opt_df = intraday_ohlc(option_dir + symbol_name + '.csv', curr_date, "%Y/%m/%d %H:%M", timedelta(minutes=1))
                            buyprice = agg_opt_df.loc[curr_datetime, 'open']
                            # buyqty = int(curr_capital / (lot_size * buyprice))
                            buyqty = position_size(lot_size, target*r2r, curr_capital, max_risk)
                            qtys.append(buyqty)
                            symbol.append(symbol_name)
                            order_type.append(signal)
                            buy_time.append(curr_datetime)
                            buy_price.append(buyprice)
                            sellprice, selltime = squareoff_price_time(agg_opt_df.loc[curr_datetime:, :], buyprice, target, r2r)
                            sell_price.append(sellprice)
                            sell_time.append(selltime)
                            pnl = (buyqty * lot_size * (sellprice - buyprice)) - calculate_trade_charges(buyprice, sellprice, buyqty, lot_size)
                            # print(symbol_name, buyprice, sellprice, curr_capital, pnl)
                            pnls.append(pnl)
                            # print(curr_capital, buyqty)
                            curr_capital = curr_capital + pnl
                            trading_capital.append(curr_capital)
                            pc_returns.append(((sellprice - buyprice) / buyprice) * 100)
                        else:
                            continue
    if len(pnls) > 0:
        overall_roi = ((curr_capital - starting_capital) / starting_capital) * 100
        overall_pc_profitability = ((len(list(filter(lambda x: (x >= 0), pnls)))) / len(pnls)) * 100
        if overall_roi >= 0:
            print("Total trades: {}, Trading days: {}, Trades/day: {:.2f}, Target: {} pts, R2R: {}, ROI: {:.2f}%, Day: {}, Profitability: {:.2f}%, Final Capital: {:.2f}".format(len(pnls), trading_days, (len(pnls)/trading_days), target, r2r, overall_roi, trade_only_on[0], overall_pc_profitability, curr_capital))
    return None

def backtest_without_target(from_date, to_date, stop_loss, max_risk, starting_capital=100000, trade_only_on=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]):
    path_lookup = pd.read_csv(expiries_lookup)
    path_lookup['Expiry'] = pd.to_datetime(path_lookup['Expiry'], format="%d/%m/%Y")
    trading_days = 0
    backtested_trades = []
    curr_capital = starting_capital
    for i in tqdm(range((to_date - from_date).days + 1)):
        curr_date = from_date + timedelta(days=i)
        expiry_date = nearest_expiry(curr_date, path_lookup)
        day_of_week = curr_date.weekday()
        # if expiry_date not in data_not_available:
        if weekday_idx[day_of_week] in trade_only_on:
            spot_file_path = path_lookup[path_lookup['Expiry'] == expiry_date]['Spot Path'].to_list()[0]
            # agg_spot_df = intraday_ohlc(spot_file_path, curr_date, "%Y%m%d %H:%M", timedelta(minutes=3))
            agg_spot_df = intraday_ohlc(spot_file_path.replace('NIFTY.csv', 'NIFTY_F1.csv'), curr_date, "%Y%m%d %H:%M", timedelta(minutes=3))
            # agg_spot_df.loc[:, ['col1', 'col2']] = agg_fut_df[['col1', 'col2']]
            if len(agg_spot_df) > 0:
                trading_days += 1
            prev_signal = None
            for curr_datetime in agg_spot_df.index:
                curr_agg_spot_df = agg_spot_df.loc[curr_datetime - timedelta(minutes=52*3): curr_datetime, :].reset_index()
                if len(curr_agg_spot_df) < 52:
                    continue
                else:
                    # print(curr_capital)
                    if curr_capital > 0: # check the correct position for this condition
                        option_dir = path_lookup[path_lookup['Expiry'] == expiry_date]['Path'].to_list()[0]
                        cmp = agg_spot_df.loc[curr_datetime, 'open']
                        atm_strike_price = atm_strike(cmp)
                        atm_strike_ce, atm_strike_pe = atm_strike_price - 0, atm_strike_price + 0
                        signal = trend_foll(curr_agg_spot_df, prev_signal)
                        if signal == 'LONG':
                            # buy ATM CE at close, log trade
                            if '__PLACEHOLDER_1__' in option_dir:
                                symbol_name = str(atm_strike_ce) + 'CE'
                                option_dir = option_dir.replace('__PLACEHOLDER_1__', symbol_name)
                            else:
                                symbol_name = 'CE ' + str(atm_strike_ce)
                                option_dir = option_dir.replace('__PLACEHOLDER_2__', symbol_name)
                            agg_opt_df = intraday_ohlc(option_dir, curr_date, "%Y/%m/%d %H:%M", timedelta(minutes=1))
                            buyprice = agg_opt_df.loc[curr_datetime, 'open']
                            buyqty = position_size(lot_size, stop_loss, curr_capital, max_risk)
                            # buyqty = 5
                            prev_signal = 'CARRYING_LONG'
                        elif signal == 'SHORT':
                            # buy ATM PE, log trade
                            if '__PLACEHOLDER_1__' in option_dir:
                                symbol_name = str(atm_strike_pe) + 'PE'
                                option_dir = option_dir.replace('__PLACEHOLDER_1__', symbol_name)
                            else:
                                symbol_name = 'PE ' + str(atm_strike_pe)
                                option_dir = option_dir.replace('__PLACEHOLDER_2__', symbol_name)
                            agg_opt_df = intraday_ohlc(option_dir, curr_date, "%Y/%m/%d %H:%M", timedelta(minutes=1))
                            buyprice = agg_opt_df.loc[curr_datetime, 'open']
                            buyqty = position_size(lot_size, stop_loss, curr_capital, max_risk)
                            # buyqty = 5
                            prev_signal = 'CARRYING_SHORT'
                        elif signal == 'EXIT_LONG' or signal == 'EXIT_SHORT':
                            if '__PLACEHOLDER_1__' in option_dir:
                                option_dir = option_dir.replace('__PLACEHOLDER_1__', symbol_name)
                            else:
                                option_dir = option_dir.replace('__PLACEHOLDER_2__', symbol_name)
                            agg_opt_df = intraday_ohlc(option_dir, curr_date, "%Y/%m/%d %H:%M", timedelta(minutes=1))
                            sellprice = agg_opt_df.loc[curr_datetime, 'open']
                            pnl = round(((sellprice-buyprice)*buyqty*lot_size) - calculate_trade_charges(buyprice, sellprice, buyqty, lot_size), 2)
                            curr_capital = curr_capital + pnl
                            backtested_trades.append([symbol_name, curr_date, weekday_idx[day_of_week], buyprice, sellprice, buyqty, pnl, curr_capital])
                            prev_signal = None
    bt_df = pd.DataFrame(backtested_trades, columns=['symbol', 'date', 'weekday', 'buy_price', 'sell_price', 'qty', 'pnl', 'curr_capital'])
    bt_result_dict = analyse_bt_result(bt_df, starting_capital, curr_capital)
    return bt_df, bt_result_dict

def analyse_bt_result(bt_df, starting_capital, curr_capital):
    '''
    Plot equity curve
    Wins, losses, max_profit, max_loss, prof_streak, loss_streak, avg_profit, avg_loss, max_dd, max_dd_pc, roi, profitability, per day stats, day wise stats
    TODO: Expectancy
    '''
    # calculate roi & prof %
    roi = round(((curr_capital - starting_capital) / starting_capital) * 100, 2)
    profitability = round(((len(list(filter(lambda x: (x >= 0), bt_df['pnl'].to_list())))) / len(bt_df['pnl'].to_list())) * 100, 2)
    bt_df = streaks(bt_df, 'pnl')

    # calculate drawdown
    prev_high = 0
    for i, cumulative in bt_df[['curr_capital']].itertuples():
        prev_high = max(prev_high, cumulative)
        dd = round(cumulative - prev_high, 2)
        dd_pc = round(((cumulative - prev_high) / prev_high) * 100, 2)
        bt_df.loc[i, 'drawdown'] = dd if dd < 0 else 0
        bt_df.loc[i, 'drawdown_pc'] = dd_pc if dd < 0 else 0

    # calculate other stats
    wins, losses, max_profit, max_loss, prof_streak, loss_streak, avg_profit, avg_loss, max_dd, max_dd_pc = len(bt_df[bt_df['pnl']>=0]), len(bt_df[bt_df['pnl']<0]), bt_df['pnl'].max(), bt_df['pnl'].min(), int(bt_df['p_streak'].max()), int(bt_df['l_streak'].max()), round(bt_df[bt_df['p_streak']>0]['pnl'].mean(), 2), round(bt_df[bt_df['l_streak']>0]['pnl'].mean(), 2), bt_df['drawdown'].min(), bt_df['drawdown_pc'].min()

    # groupby date
    date_grouped = bt_df.groupby(['date']).agg(Trading_Capital = ('curr_capital', lambda x: list(x)[-1]), Daily_PnL = ('pnl', sum)).reset_index()
    date_grouped.loc[-1] = [date_grouped.loc[0, "date"], starting_capital, 0]
    date_grouped.index = date_grouped.index + 1
    date_grouped.sort_index(inplace=True)

    # per day stats
    max_profit_per_day, max_loss_per_day, avg_profit_per_day, avg_loss_per_day = round(date_grouped['Daily_PnL'].max(), 2), round(date_grouped['Daily_PnL'].min(), 2), round(date_grouped[date_grouped['Daily_PnL'] >= 0]['Daily_PnL'].mean(), 2), round(date_grouped[date_grouped['Daily_PnL'] < 0]['Daily_PnL'].mean(), 2)
    stats_df = pd.DataFrame([['Profitability %', str(profitability)+' %'], ['Winning, losing trades', "{0}, {1}".format(wins, losses)], ['Winning, losing streak', "{0}, {1}".format(prof_streak, loss_streak)], ['Max Profit / trade', max_profit], ['Max Loss / trade', max_loss], ['Max Profit / day', max_profit_per_day], ['Max Loss / day', max_loss_per_day], ['Avg Profit / trade', avg_profit], ['Avg Loss / trade', avg_loss], ['Avg Profit / day', avg_profit_per_day], ['Avg Loss / day', avg_loss_per_day], ['Max DD', max_dd], ['Max DD %', str(max_dd_pc) + ' %'], ['ROI %', str(roi)+' %']])

    # groupby weekday
    weekday_grouped = bt_df.groupby(['weekday']).agg(Total_PnL = ('pnl', sum)).reset_index().round(2)
    weekday_grouped['Day'] = pd.Categorical(weekday_grouped['weekday'], categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'], ordered=True)
    weekday_grouped.sort_values(by=['Day'], inplace=True)
    weekday_grouped = weekday_grouped[['Day', 'Total_PnL']]

    # Plot the curve & tables
    FONTSIZE = 9
    fig = plt.figure()
    gs = fig.add_gridspec(2, hspace=0.5)
    ax = gs.subplots()
    tab1 = ax[1].table(cellText=weekday_grouped.values, colLabels=None, colWidths=[0.20, 0.20], loc='center left')
    tab2 = ax[1].table(cellText=stats_df.values, colLabels=None, colWidths=[0.31, 0.19], loc='center right')
    tab1.auto_set_font_size(False)
    tab1.set_fontsize(FONTSIZE)
    tab2.auto_set_font_size(False)
    tab2.set_fontsize(FONTSIZE)
    ax[1].get_xaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)
    ax[0].plot(date_grouped['date'], date_grouped['Trading_Capital'])
    ax[0].axhline(y=starting_capital, color = 'r', linestyle='dashed')
    plt.box(on=None)
    plt.show()
    result_dict =  {}
    return result_dict

def streaks(df, col):
    sign = np.sign(df[col])
    s = sign.groupby((sign!=sign.shift()).cumsum()).cumsum()
    return df.assign(p_streak=s.where(s>0, 0), l_streak=s.where(s<0, 0).abs())

if __name__ == '__main__':
    '''
    TODO: Day wise. Daily. Monthly
    '''
    from_date = datetime(2019, 2, 11)
    # from_date = datetime(2021, 4, 5)
    to_date = datetime(2021, 4, 22)
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    ##############################        backtesting with target          ###############################
    # tgt_pts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # r2rs = [0.5, 0.667, 0.75, 1, 1.25, 1.33, 1.5, 1.75, 2, 2.25, 2.5, 3]

    # for pt in tgt_pts:
    #     for r2r in r2rs:
    #         for week_day in weekdays:
    #             backtest(from_date, to_date, pt, r2r, 0.05, trade_only_on=[week_day])

    # for day_of_week in weekdays:
    #     backtest(from_date, to_date, 10, 0.5, 0.05, trade_only_on=[day_of_week])

    # backtest(from_date, to_date, 7, 0.5, 0.05, trade_only_on=["Thursday"])

    ##############################        backtesting without target       ###############################
    stop_losses = [5, 7, 10, 12, 15, 20]

    # for stop_loss in stop_losses:
    #     for day_of_week in weekdays:
    #         backtest_without_target(from_date, to_date, stop_loss, 0.05, trade_only_on=[day_of_week])

    # for day_of_week in weekdays:
    #     backtest_without_target(from_date, to_date, 20, 0.05, trade_only_on=[day_of_week])

    backtest_without_target(from_date, to_date, 20, 0.02, trade_only_on=["Monday", "Tuesday", "Wednesday", "Thursday"])
    # backtest_without_target(from_date, to_date, 5, 0.05)