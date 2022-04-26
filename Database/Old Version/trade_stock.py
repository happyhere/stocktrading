import sys
import os 
import time
import pandas as pd
import datetime
import telegram
# Scheduler
# from apscheduler.schedulers.blocking import BlockingScheduler
# from pytz import utc,timezone
# Investpy
import investpy
#import asyncio
import requests
from user import USER_AGENTS
from pandas import json_normalize
import random
# TA-lib: https://pypi.org/project/TA-Lib/
from talib import abstract
# import plotly.express as px
import plotly.graph_objects as go
from scipy import stats, signal
import numpy as np

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
CACHE_PATH = DIR_PATH+'/cache'
# Check whether the specified path exists or not
isExist = os.path.exists(CACHE_PATH)
if not isExist:
  # Create a new directory because it does not exist 
  os.makedirs(CACHE_PATH)

# Dump print to log file
old_stdout = sys.stdout
LOG_FILE = open(CACHE_PATH + "/trade_stock.log","a")
sys.stdout = LOG_FILE

MODE = "VND" # VND/SSI/TCB OR INVESTPY
TELEGRAM_API_ID = "5030826661:AAH-C7ZGJexK3SkXIqM8CyDgieoR6ZFnXq8"
TELEGRAM_CHANNEL_ID = "@botmuabanchungkhoan"
# TIME_ZONE = timezone("Asia/Ho_Chi_Minh")
STOCK_FILE = "Database/ToTheMoon.tls"

NEXT_TIME_FILE = CACHE_PATH + "/NextTimeFile-Stock.txt"
START_TRADE_TIME_ORIGINAL = datetime.datetime.strptime("2022-04-01",'%Y-%m-%d') #GMT+7 Trade time count if day > 15h else day -= 1
TIME_INTERVAL_DELTA = datetime.timedelta(days = 1) # Write next time search
TIME_DURATION_DELTA = datetime.timedelta(days = 366)
TIME_PROTECT_DELTA = datetime.timedelta(hours = 15, minutes= 10) # Add 15 hours 10 minutes to prevent missing data for 1 day interval 

#Start trade cannot be Sunday, Saturday
if START_TRADE_TIME_ORIGINAL.weekday() == 5:
  print(" START_TRADE_TIME_ORIGINAL is Saturday")
  START_TRADE_TIME = START_TRADE_TIME_ORIGINAL - TIME_INTERVAL_DELTA
elif START_TRADE_TIME_ORIGINAL.weekday() == 6:
  print("START_TRADE_TIME_ORIGINAL is Sunday")
  START_TRADE_TIME = START_TRADE_TIME_ORIGINAL - 2*TIME_INTERVAL_DELTA
else:
  START_TRADE_TIME = START_TRADE_TIME_ORIGINAL

# # Scheduler for any plans
# scheduler = BlockingScheduler()

### https://github.com/thinh-vu/vnstock/blob/main/vnstock/stock.py
def stock_historical_data (symbol, start_date, end_date):
    """
    This function returns the stock historical daily data.
    Args:
        symbol (:obj:`str`, required): 3 digits name of the desired stock.
        start_date (:obj:`str`, required): the start date to get data (YYYY-mm-dd).
        end_date (:obj:`str`, required): the end date to get data (YYYY-mm-dd).
    Returns:
        :obj:`pandas.DataFrame`:
        | tradingDate | open | high | low | close | volume |
        | ----------- | ---- | ---- | --- | ----- | ------ |
        | YYYY-mm-dd  | xxxx | xxxx | xxx | xxxxx | xxxxxx |
    Raises:
        ValueError: raised whenever any of the introduced arguments is not valid.
    """ 
    fd = int(time.mktime(time.strptime(start_date, "%Y-%m-%d")))
    td = int(time.mktime(time.strptime(end_date, "%Y-%m-%d")))
    data = requests.get('https://apipubaws.tcbs.com.vn/stock-insight/v1/stock/bars-long-term?ticker={}&type=stock&resolution=D&from={}&to={}'.format(symbol, fd, td)).json()
    df = json_normalize(data['data'])
    # df['tradingDate'] = pd.to_datetime(df.tradingDate.str.split("T", expand=True)[0])
    # df.columns = df.columns.str.title()
    # df.rename(columns={'Tradingdate':'TradingDate'}, inplace=True)
    return df


### https://github.com/thanhtlx/api-stock/blob/main/apiStock/core.py
def DICT_FILTER(x, y): return dict([(i, x[i]) for i in x if i in set(y)])
KEYS = ['date', 'adOpen',
        'adHigh', 'adLow', 'adClose',
        "nmVolume"]

def getStockHistory(code, start_date, end_date):
    try:
        URL = 'https://finfo-api.vndirect.com.vn/v4/stock_prices?sort=date&q=code:%s~date:gte:%s~date:lte:%s&size=100000'
        r = requests.get(URL % (code, start_date, end_date),
                         headers={"USER-AGENT": USER_AGENTS[random.randint(0, len(USER_AGENTS)-1)]})
        res = r.json()
    except Exception as e:
        return []
    if 'data' in res and len(res['data']) > 0:
        res = list(map(lambda x: DICT_FILTER(x, KEYS), res['data']))
        res.reverse()
        df =  pd.DataFrame(res) 
        df.rename(columns={'date': 'timestamp', 'adOpen': 'open', 'adHigh': 'high', 'adLow': 'low', 'adClose': 'close', 'nmVolume': 'volume'}, inplace=True)
        # print(df)
        return df
    return []

## STOCK TRADING HISTORICAL DATA
def getStockHistoryV2(code,start_date,end_date):
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    end_time = int(end_date.timestamp())
    start_time = int(start_date.timestamp())
    try:
        URL_SSI = "https://iboard.ssi.com.vn/dchart/api/history?resolution=D&symbol=%s&from=%s&to=%s" 
        r = requests.get(URL_SSI % (code, start_time, end_time),
                         headers={"USER-AGENT": USER_AGENTS[random.randint(0, len(USER_AGENTS)-1)]})
        res = r.json()
    except Exception as e:
        return []
    result = dict()
    # oriHeader = ['t', 'o', 'h', 'l', 'c', 'v']
    # header = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
    for k,v in res.copy().items():
        if isinstance(v, list):
            v.reverse()
            result[k] = v
    df =  pd.DataFrame.from_dict(result) 
    df.rename(columns={'t': 'timestamp', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}, inplace=True)
    # print(convert_timestamp_to_date(df["Timestamp"][0]))
    # df["Timestamp"]=pd.to_pydatetime(df["Timestamp"])
    # df["Timestamp"]=df["Timestamp"].dt.strftime("%Y-%m-%d")
    df['timestamp'] = df['timestamp'].apply(lambda x: convert_date_to_string(convert_timestamp_to_date(x)))
    df['open'] = df['open'].apply(lambda x: float(x))
    df['high'] = df['high'].apply(lambda x: float(x))
    df['low'] = df['low'].apply(lambda x: float(x))
    df['close'] = df['close'].apply(lambda x: float(x))
    df['volume'] = df['volume'].apply(lambda x: float(x))
    # print(df)
    return df

def convert_timestamp_to_date(timeStampStr):
  return datetime.datetime.fromtimestamp(int(timeStampStr))

if MODE == "VND":
  def convert_to_datetime(stockData):
    return datetime.datetime.strptime(stockData["timestamp"],'%Y-%m-%d')
  def convert_date_to_string(dateTime):
    return dateTime.strftime('%Y-%m-%d')
elif MODE == "INVESTPY":
  def convert_to_datetime(stockData):
    return stockData.name
  def convert_date_to_string(dateTime):
    return dateTime.strftime('%d/%m/%Y')

def caculate_start_index(stock_data):
  for index in range (1, (len(stock_data))):
    tmpTime = convert_to_datetime(stock_data.iloc[index])
    if tmpTime > (START_TRADE_TIME-TIME_INTERVAL_DELTA):
      return index

  return len(stock_data) # Mean don't analysis anything

def analysis_stock(stock_name, stock_data, nextTimeStamp):
  """ Analysis one stock
  """
  # Caculate indicator
  volumeProfiles = caculate_volume_profiles(stock_data)
  stock_data["ma10"] = ma10 = abstract.SMA(stock_data, timeperiod=10)
  stock_data["ma20"] = ma20 = abstract.SMA(stock_data, timeperiod=20)
  rsi = abstract.RSI(stock_data, timeperiod=14)
  macd_histogram = abstract.MACD(stock_data).macdhist
  change = abstract.ROC(stock_data)
  close = stock_data["close"]
  high = stock_data["high"]
  low = stock_data["low"]
  atr = abstract.ATR(stock_data, timeperiod=10)
  invertHammer = abstract.CDLINVERTEDHAMMER(stock_data)

  buyPrice = 0
  hold = 0
  stoplossPrice = 0
  index  = caculate_start_index(stock_data)
  print("#" + stock_name + " Day: " , nextTimeStamp)
  for i in range(index, len(stock_data)): # Crypto: don't use last value
      commands = [] # 0: Do nothing, 1: Buy, 2: Sell, 3-4: Stoploss, 5: Increase stoploss, 6: Buy if missing, 7-11: Indicators 
      # If the MA10 crosses MA20 line upward
      if ma10[i] > ma20[i] and ma10[i - 1] <= ma20[i - 1] and macd_histogram[i] > -0.1:
        commands.append(1)
        buyPrice = close[i]
        hold = 1
        if (atr[i] > 0.085*buyPrice): # Stoploss should start from 8.5%
          stoplossPrice = buyPrice - atr[i]
        else:
          stoplossPrice = buyPrice*0.915

      # Stoploss steal your balance
      if low[i] <= stoplossPrice and hold == 1 and stoplossPrice != 0 and len(commands) == 0:
        commands.append(3)
        hold = 0

      # The other way around
      if ma10[i] < ma20[i] and ma10[i - 1] >= ma20[i - 1] and hold == 1 and len(commands) == 0:
        commands.append(2)
        hold = 0
        stoplossPrice = 0

      # Increase stoploss to protect balance
      if close[i] >= (2*buyPrice-stoplossPrice) and hold == 1 and stoplossPrice != buyPrice and len(commands) == 0:
        commands.append(4)
        stoplossPrice = buyPrice

      # Buy if have a chance
      if close[i] <= buyPrice and hold == 1 and len(commands) == 0:
        commands.append(5)
      
      # Cross RSI, MCDA, MA, 8.5% change OR ((H-C)/C > 0.1)
      if rsi[i] < 70 and rsi[i - 1] >= 70 and hold == 1 and (len(commands) == 0 or commands[0] > 3):
        commands.append(6)

      if macd_histogram[i] < 0 and macd_histogram[i - 1] >= 0 and hold == 1 and (len(commands) == 0 or commands[0] > 3):
        commands.append(7)

      if change[i] <= -5 and hold == 1 and (len(commands) == 0 or commands[0] > 3):
        commands.append(8)

      if invertHammer[i] != 0 and hold == 1 and (len(commands) == 0 or commands[0] > 3):
        commands.append(9)
      
      if len(volumeProfiles) > 0 and hold == 1 and (len(commands) == 0 or commands[0] > 3):
        for key in volumeProfiles:
          diff = abs(close[i]-key)/close[i]
          if diff < 0.03: # Compare to resistance/support with 5%
            commands.append(10)
            break

      if (nextTimeStamp - TIME_INTERVAL_DELTA) < convert_to_datetime(stock_data.iloc[i]):
        print ("Processing: ", stock_name, i, convert_to_datetime(stock_data.iloc[i]))
        # Push to telegram
        if (len(commands) > 0):
          message = prepare_message(stock_name, commands, stock_data.iloc[i], buyPrice, volumeProfiles, stoplossPrice)
          send_message_telegram(message)
      elif i == (len(stock_data)-1) and (nextTimeStamp - TIME_INTERVAL_DELTA) >= convert_to_datetime(stock_data.iloc[i]):
        # Time is comming but no stock data fitting
        raise Exception("Check the system, " + stock_name + " does not have next data: " + 
          convert_date_to_string(convert_to_datetime(stock_data.iloc[i])))

# Prepare message to send anywhere
def prepare_message(stock_name, commands, stock_data, buyPrice, volumeProfiles, stoplossPrice):
  message = " #" + stock_name + " Day: " + convert_to_datetime(stock_data).strftime("%d, %b %Y")
  profit_report = 0
  stoploss_setting = 0
  zone_setting = 0
  for command in commands:
    match command:
      case 1: # Buy: +0.2%
        stoploss_setting = 1
        zone_setting = 1
        message += "\n" + " - Buy at: {:.3f}".format(stock_data["close"]*1.002)
      case 2: # Sell: 0.4~0.5%
        profit_report = 1
        message += "\n" + " - Sell at: {:.3f}".format(stock_data["close"]*0.998)
      case 3:
        message += "\n" + " - Stoploss steal {:.2f}% at: {:.3f}".format((stoplossPrice-buyPrice)*100/buyPrice,stoplossPrice)
      case 4:
        message += "\n" + " - Stoploss should set at: " + "{:.3f}".format(buyPrice)
      case 5: # Buy: +0.2%
        stoploss_setting = 1
        zone_setting = 1
        message += "\n" + " - Possible Risk Buy, Old at: {:.3f}".format(buyPrice)
      case 6:
        profit_report = 1
        message += "\n" + " - RSI cross-down below 70"
      case 7:
        profit_report = 1
        message += "\n" + " - Histogram cross-down below 0"
      case 8:
        profit_report = 1
        message += "\n" + " - Bar price drop-down below 5%"
      case 9:
        profit_report = 1
        message += "\n" + " - Bar is invert-Hammer check balance"
      case 10:
        zone_setting = 1
        message += "\n" + " - Near the support/resistance zone"
  
  if (stoploss_setting == 1):
    message += "\n" + " - Stoploss at : {:.3f} {:.2f}%".format(stoplossPrice, ((stoplossPrice/buyPrice)-1)*100); # Stoploss > 8.5%
  if (profit_report == 1 and stoploss_setting == 0):
    message += "\n" + " - Profit at: {:.2f}%".format((stock_data["close"]-buyPrice)/buyPrice*100) #*0.996
  if (zone_setting == 1):
    for key in volumeProfiles:
      message += "\n" + " - Zone {:.3f}: {:.2f}M".format(key, volumeProfiles[key])

  message += "\n" + " - Close at : " + "{:.3f}".format(stock_data["close"]) \
			    +  "\n" + " - Value at MA10: " + "{:.3f}".format(stock_data['ma10']) \
			    +  "\n" + " - Value at MA20: " + "{:.3f}".format(stock_data['ma20'])

  return message

def send_message_telegram(message):
  try:
    telegram_notify = telegram.Bot(TELEGRAM_API_ID)
    telegram_notify.send_message(chat_id=TELEGRAM_CHANNEL_ID, text=message,
                            parse_mode='Markdown')
    print("Sent a message to Telegram")
    time.sleep(1)
  except Exception as ex:
    print(ex)

def get_dist_plot(c, v, kx, ky):
    fig = go.Figure()
    fig.add_trace(go.Histogram(name='Vol Profile', x=c, y=v, nbinsx=150, 
                               histfunc='sum', histnorm='probability density',
                               marker_color='#B0C4DE'))
    fig.add_trace(go.Scatter(name='KDE', x=kx, y=ky, mode='lines', marker_color='#D2691E'))
    return fig

def caculate_volume_profiles(stock_data):
    # https://medium.com/swlh/how-to-analyze-volume-profiles-with-python-3166bb10ff24
    # print(stock_data["close"].min(), " ", stock_data["close"].max(), " ", len(stock_data["close"]))
    kde_factor = 0.05
    num_samples = len(stock_data)
    kde = stats.gaussian_kde(stock_data["close"],weights=stock_data["volume"],bw_method=kde_factor)
    xr = np.linspace(stock_data["close"].min(),stock_data["close"].max(),num_samples)
    kdy = kde(xr)

    # Find peaks
    min_prom = kdy.max() * 0.2
    peaks, peak_props = signal.find_peaks(kdy, prominence=min_prom)

    # Add peaks to dictionary
    volumePeaks = {}
    # pocIndex = -1
    for i in range(0, len(peaks)):
      volumePeaks[xr[peaks[i]]] = kdy[peaks[i]]/kdy.min()*100
    #   if kdy[peaks[pocIndex]] < kdy[peaks[i]]:
    #     pocIndex = i
    # if pocIndex >= 0:
    #   volumePeaks["poc"] = xr[peaks[pocIndex]]
    # print (volumePeaks)

    ##### Draw the figure
    # ticks_per_sample = (xr.max() - xr.min()) / num_samples
    # pk_marker_args=dict(size=10)
    # pkx = xr[peaks]
    # pky = kdy[peaks]

    # fig = get_dist_plot(stock_data["close"], stock_data["volume"], xr, kdy)
    # fig.add_trace(go.Scatter(name='Peaks', x=pkx, y=pky, mode='markers', marker=pk_marker_args))
    # fig.show()
    return volumePeaks

def read_stock_list():
  stockFile = open(STOCK_FILE, "r")
  stockList = []
  for stockLine in stockFile:
    stockName = stockLine.split("\n")
    stockList.append(stockName[0])
  stockFile.close()
  # print(stockList)
  
  return stockList

# Read/Write format: %d-%m-%Y
def read_next_time_stamp():
  try:
    with open(NEXT_TIME_FILE, "r") as file:
        return datetime.datetime.strptime(file.read(),'%d-%m-%Y')
  except IOError:
    return START_TRADE_TIME

def write_next_time_stamp(nextTimeStamp):
  with open(NEXT_TIME_FILE, "w+") as file:
    file.write(nextTimeStamp.strftime('%d-%m-%Y'))

if MODE == "VND":
  def verify_stock_data(stockName, startTime, endTime): # Check getting from VN Direct
    MAXIMUM_RETRY = 5 # Retry maxium 5 times if getting data failure
    retryCount = 0
    while(retryCount < MAXIMUM_RETRY): 
      stockData = getStockHistory(stockName, startTime, endTime)
      if any("close" in s for s in stockData):
        break
      else: # not stockData
        print("Request stock data empty count:", retryCount)
        retryCount += 1
        time.sleep(1)

      if (retryCount == MAXIMUM_RETRY):
        raise Exception("Getting stock data: " + stockName + " failure")
    return stockData
elif MODE == "INVESTPY":
  def verify_stock_data(stockName, startTime, endTime): # Check getting from Investpy
    stockData = investpy.get_stock_historical_data(stock=stockName, country='VietNam', from_date=startTime, to_date=endTime)
    stockData.drop('Currency', axis=1, inplace=True)
    stockData.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
    # df.index.name = 'timestamp'
    return stockData

def schedule_analysis_stock():
  currentTime = datetime.datetime.now()
  print("------ Day:", currentTime, "------")
  stockList = read_stock_list()
  # Read nextTimeStamp from file otherwise default: START_TRADE_TIME
  nextTimeStamp = read_next_time_stamp()
  if currentTime >= (nextTimeStamp+TIME_PROTECT_DELTA):
    try:
      send_message_telegram("-------------- Day: " + currentTime.strftime("%d, %b %Y") + " --------------")
      for stockName in stockList:
        # print("Pair: " + stockName)
        stock_data = verify_stock_data(stockName, convert_date_to_string(currentTime-TIME_DURATION_DELTA), convert_date_to_string(currentTime))
        analysis_stock(stockName, stock_data, nextTimeStamp)

      lastTimeStamp = convert_to_datetime(stock_data.iloc[-1])
      if lastTimeStamp.weekday() == 4: # Friday and next time is Monday
        nextTimeStamp = lastTimeStamp + 3*TIME_INTERVAL_DELTA
      else:
        nextTimeStamp = lastTimeStamp + TIME_INTERVAL_DELTA
      write_next_time_stamp(nextTimeStamp)
      print ("------ End of Day:", currentTime, "------")
      send_message_telegram("---------- End of Day: " + currentTime.strftime("%d, %b %Y") + " ----------")
    except Exception as ex:
      send_message_telegram("Something wrong: " + str(ex))
      print("Something wrong: ", ex)
      # print(stock_data)
      # scheduler.shutdown()
  else:
    print("Please run again after 15pm, next time is", nextTimeStamp+TIME_PROTECT_DELTA)

if __name__ == "__main__":
  try:
    # Run first time if needed
    schedule_analysis_stock()

    # Program ended, turn off sys log file mode
    sys.stdout = old_stdout
    LOG_FILE.close()

  except Exception as ex:
    print("Program Exception: Is it related Telegram?", ex)
    # Program ended, turn off sys log file mode
    sys.stdout = old_stdout
    LOG_FILE.close()

  # Windows sleep this task so using Window Task Schedule
  # scheduler.add_job(schedule_analysis_stock, 'cron', day_of_week="mon-fri", hour="15", minute="30", timezone=TIME_ZONE, misfire_grace_time=None)  # run on Monday to Friday at 15h30
  # try:
  #     scheduler.start()
  # except (KeyboardInterrupt, SystemExit):
  #     pass
  