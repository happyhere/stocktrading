import sys
import os 
import time
import pandas as pd
import datetime
import telegram
# Scheduler
# from apscheduler.schedulers.blocking import BlockingScheduler
# from pytz import utc,timezone
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
import argparse

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
CACHE_PATH = DIR_PATH+'/cache'
# Check whether the specified path exists or not
isExist = os.path.exists(CACHE_PATH)
if not isExist:
  # Create a new directory because it does not exist 
  os.makedirs(CACHE_PATH)

parser = argparse.ArgumentParser()
parser.add_argument("-hm", "--historical-mode", help="Historical mode to select which vendor getting historical data "
                                                     "Vendor: VND/SSI/TCB #Reserved for future dev ",
                    default="VND", type=str)
parser.add_argument("-s", "--server-mode", help="Run program in server like pythonanywhere/heroku "
                                                "by default program will run in local machine",
                    action="store_true", default=False)
args = parser.parse_args()

SERVER_MODE = args.server_mode # Heroku Server: GMT +0, print log
HISTORICAL_MODE = args.historical_mode
TELEGRAM_API_ID = "5030826661:AAH-C7ZGJexK3SkXIqM8CyDgieoR6ZFnXq8"
TELEGRAM_CHANNEL_ID = "@botmuabanchungkhoan"
# TIME_ZONE = timezone("Asia/Ho_Chi_Minh")
STOCK_FILE = DIR_PATH+"/Database/ToTheMoon.tls"

NEXT_TIME_FILE = CACHE_PATH + "/NextTimeFile-Stock.txt"
START_TRADE_TIME_ORIGINAL = datetime.datetime.strptime("2022-04-01",'%Y-%m-%d') #GMT+7 Trade time count if day > 15h else day -= 1
TIME_INTERVAL_DELTA = datetime.timedelta(days = 1) # Write next time search
TIME_DURATION_DELTA = datetime.timedelta(days = 366)
TIME_PROTECT_DELTA = datetime.timedelta(hours = 15, minutes= 10) # Add 15 hours 10 minutes to prevent missing data for 1 day interval 

if not SERVER_MODE:
  TIME_UTC_DELTA = datetime.timedelta(hours = 0)
  # Dump print to log file
  old_stdout = sys.stdout
  LOG_FILE = open(CACHE_PATH + "/trade_stock.log","a")
  sys.stdout = LOG_FILE
else:
  TIME_UTC_DELTA = datetime.timedelta(hours = 7)

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

def convert_to_datetime(stockData):
  return datetime.datetime.strptime(stockData["timestamp"],'%Y-%m-%d')

def convert_date_to_string(dateTime):
  return dateTime.strftime('%Y-%m-%d')

def send_message_telegram(message):
  try:
    telegram_notify = telegram.Bot(TELEGRAM_API_ID)
    telegram_notify.send_message(chat_id=TELEGRAM_CHANNEL_ID, text=message,
                            parse_mode='Markdown')
    print("Sent a message to Telegram")
    time.sleep(1)
  except Exception as ex:
    print(ex)

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
if not SERVER_MODE:
  def read_next_time_stamp():
    try:
      with open(NEXT_TIME_FILE, "r") as file:
          return datetime.datetime.strptime(file.read(),'%d-%m-%Y')
    except IOError:
      return START_TRADE_TIME
else:
  def read_next_time_stamp():
    try:
      nextTimeStamp = os.environ['NEXT_TIME_STAMP_STOCK']
      return datetime.datetime.strptime(nextTimeStamp,'%d-%m-%Y')
    except KeyError:
      try:
        with open(NEXT_TIME_FILE, "r") as file:
            return datetime.datetime.strptime(file.read(),'%d-%m-%Y')
      except IOError:
        return START_TRADE_TIME

def write_next_time_stamp(nextTimeStamp):
  with open(NEXT_TIME_FILE, "w+") as file:
    file.write(nextTimeStamp.strftime('%d-%m-%Y'))

class Trade:
  def __init__(self, stockName, currentTime, nextTimeStamp):
    self.stockName = stockName
    self.currentTime = currentTime
    self.nextTimeStamp = nextTimeStamp
    self.stockData = []
    self.dataLength = 0
    self.startIndex = 0
    self.volume = []
    self.close = []
    self.high = []
    self.low = []
    self.volumeProfiles = []
    self.ma5 = []
    self.ma10 = []
    self.ma20 = []
    self.rsi = []
    self.macdHistogram = []
    self.change = []
    self.atr = []
    # shooting star, hanging man, evening star, Marubozu < 0, CDLADVANCEBLOCK, CDLHARAMI < 0, CDLHIGHWAVE < 0
    self.shootingStar = []
    self.hangingMan = []
    self.eveningStar = []
    self.marubozu = []
    self.advanceBlock = []
    self.harami = []
    self.highWave = []
    # Process time by time
    self.refPrice = 0
    self.buyPrice = 0
    self.hold = 0
    self.stoplossPrice = 0
    self.commands = []
    self.processIndex = 0
    self.message = ""
    self.candlestick = ""
    self.verify_stock_data()
    self.caculate_start_index()
    self.get_stock_indicators()

  def verify_stock_data(self): # Check getting from VN Direct
    MAXIMUM_RETRY = 5 # Retry maxium 5 times if getting data failure
    retryCount = 0
    while(retryCount < MAXIMUM_RETRY): 
      stockData = getStockHistory(self.stockName, convert_date_to_string(self.currentTime-TIME_DURATION_DELTA), 
                                  convert_date_to_string(self.currentTime))
      if any("close" in s for s in stockData):
        break
      else: # not stockData
        print("Request stock data empty count:", retryCount)
        retryCount += 1
        time.sleep(1)

      if (retryCount == MAXIMUM_RETRY):
        print(stockData)
        raise Exception("Getting stock data: " + self.stockName + " failure")
    self.stockData = stockData
    self.dataLength = len(stockData)

  def caculate_start_index(self):
    for index in range (1, (self.dataLength)):
      tmpTime = convert_to_datetime(self.stockData.iloc[index])
      if tmpTime > (START_TRADE_TIME-TIME_INTERVAL_DELTA):
        self.startIndex = index
        return
    self.startIndex = self.dataLength # Mean don't analysis anything

  def get_stock_indicators(self):
    # Caculate indicator
    self.close = self.stockData["close"]
    self.high = self.stockData["high"]
    self.low = self.stockData["low"]
    self.volume = self.stockData["volume"]
    self.volumeProfiles = self.caculate_volume_profiles()
    self.ma5 = abstract.SMA(self.stockData, timeperiod=5)
    self.ma10 = abstract.SMA(self.stockData, timeperiod=10)
    self.ma20 = abstract.SMA(self.stockData, timeperiod=20)
    self.rsi = abstract.RSI(self.stockData, timeperiod=14)
    self.macdHistogram = abstract.MACD(self.stockData).macdhist
    self.change = abstract.ROC(self.stockData, timeperiod=1)
    self.atr = abstract.ATR(self.stockData, timeperiod=10)
    self.shootingStar = abstract.CDLSHOOTINGSTAR(self.stockData)
    self.hangingMan = abstract.CDLHANGINGMAN(self.stockData)
    self.eveningStar = abstract.CDLEVENINGSTAR(self.stockData)
    self.marubozu = abstract.CDLMARUBOZU(self.stockData)
    self.advanceBlock = abstract.CDLADVANCEBLOCK(self.stockData)
    self.harami = abstract.CDLHARAMI(self.stockData)
    self.highWave = abstract.CDLHIGHWAVE(self.stockData)

  def analysis_stock(self):
    """ Analysis one stock
    """
    print("#" + self.stockName + " Day: " , self.nextTimeStamp)
    for i in range(self.startIndex, self.dataLength): # Crypto: don't use last value
      if i < 20: # Condition to fit MA20 has value
        continue
      self.commands = [] # 0: Do nothing, 1: Buy, 2: Sell, 3-4: Stoploss, 5: Increase stoploss, 6: Buy if missing, 7-11: Indicators
      # Caculate prediction next day MA10 vs MA20
      nextMA10 = (self.ma10[i]*10 - self.close[i-9] + self.close[i])/10
      nextMA20 = (self.ma20[i]*20 - self.close[i-19] + self.close[i])/20
      # If the MA10 crosses MA20 line upward
      if self.ma10[i] > self.ma20[i] and self.ma10[i - 1] <= self.ma20[i - 1] and self.hold == 0: # and self.macdHistogram[i] > -0.1
        self.commands.append(1)
        self.buyPrice = self.close[i]
        self.hold = 1
        refPrice = (self.ma5[i] + self.ma10[i] + self.ma20[i])/3 # Reference for buy
        if (refPrice > self.close[i]):
          self.refPrice = self.close[i]
        else:
          self.refPrice = refPrice
        self.stoplossPrice = self.buyPrice - self.atr[i] # Stoploss based ATR(10)
        # if (self.atr[i] > 0.085*self.buyPrice): # Stoploss should start from 8.5%
        #   self.stoplossPrice = self.buyPrice - self.atr[i]
        # else:
        #   self.stoplossPrice = self.buyPrice*0.915
      # The other way around
      elif self.ma10[i] < self.ma20[i] and self.ma10[i - 1] >= self.ma20[i - 1] and self.hold == 1 and len(self.commands) == 0:
        self.commands.append(2)
        self.hold = 0
        self.stoplossPrice = 0

      if self.hold == 1: 
        # Stoploss warning trigger your balance
        if self.low[i] < self.stoplossPrice and self.stoplossPrice != 0 and len(self.commands) == 0:
          self.commands.append(3)
          # self.hold = 0

        # Increase stoploss to protect balance
        if self.close[i] >= (2*self.buyPrice-self.stoplossPrice) and \
            self.stoplossPrice != self.buyPrice and (len(self.commands) == 0 or self.commands[0] > 2):
          self.commands.append(4)
          self.stoplossPrice = self.buyPrice

        # Buy if have a chance, refPrice <-> close: 7%
        if (self.close[i]*0.93) <= self.refPrice and (len(self.commands) == 0 or self.commands[0] > 2):
          self.commands.append(5)
      
      if self.hold == 1 or len(self.commands) > 0:
        # Cross RSI, MCDA, MA, -5% change
        if self.rsi[i] < 70 and self.rsi[i - 1] >= 70:
          self.commands.append(6)

        if self.macdHistogram[i] < 0 and self.macdHistogram[i - 1] >= 0:
          self.commands.append(7)

        if self.change[i] <= -5:
          self.commands.append(8)

        self.candlestick = ""
        if self.shootingStar[i] != 0:
          self.candlestick += "shootingStar "
        if self.hangingMan[i] != 0:
          self.candlestick += "hangingMan "
        if self.eveningStar[i] != 0:
          self.candlestick += "eveningStar "
        if self.marubozu[i] < 0:
          self.candlestick += "marubozu "
        if self.advanceBlock[i] != 0:
          self.candlestick += "advanceBlock "
        if self.harami[i] < 0:
          self.candlestick += "harami "
        if self.highWave[i] < 0:
          self.candlestick += "highWave "
        if self.candlestick != "":
          self.commands.append(9)
        
        if len(self.volumeProfiles) > 0:
          for key in self.volumeProfiles:
            diff = abs(self.close[i]-key)/self.close[i]
            if diff < 0.03: # Compare to resistance/support with 5%
              self.commands.append(10)
              break

        if self.ma5[i] < self.ma10[i]: #Downtrend warning
          self.commands.append(11)

        if nextMA10 < nextMA20: # Warning next downtrend
          self.commands.append(13)

      if self.hold == 0:
        if self.ma5[i] >= self.ma10[i] and len(self.commands) > 0: # Possible to hold on Sell
          self.commands.append(12)
        if nextMA10 >= nextMA20: # Warning next uptrend
          self.commands.append(14)

      if (self.nextTimeStamp - TIME_INTERVAL_DELTA) < convert_to_datetime(self.stockData.iloc[i]):
        print ("Processing: ", self.stockName, i, convert_to_datetime(self.stockData.iloc[i]))
        # Push to telegram
        if (len(self.commands) > 0):
          self.processIndex = i
          self.prepare_message()
          send_message_telegram(self.message)
      elif i == (self.dataLength-1) and (self.nextTimeStamp - TIME_INTERVAL_DELTA) >= convert_to_datetime(self.stockData.iloc[i]):
        # Time is comming but no stock data fitting
        raise Exception("Check the system, " + self.stockName + " does not have target data, last: " + 
          convert_date_to_string(convert_to_datetime(self.stockData.iloc[i])))

  # Prepare message to send anywhere
  def prepare_message(self):
    currentData = self.stockData.iloc[self.processIndex]
    message = " #" + self.stockName + " Day: " + convert_to_datetime(currentData).strftime("%d, %b %Y")
    profit_report = 0
    stoploss_setting = 0
    for command in self.commands:
      match command:
        case 1: # Buy: +0.2%
          stoploss_setting = 1
          message += "\n" + " - Ref Buy at: {:.3f}".format(self.refPrice*1.003)
        case 2: # Sell: 0.4~0.5%
          profit_report = 1
          message += "\n" + " - Sell at: {:.3f}".format(currentData["close"]*0.998)
        case 3:
          profit_report = 1
          message += "\n" + " - Stoploss reached at: {:.3f}".format(self.stoplossPrice)
        case 4:
          profit_report = 1
          message += "\n" + " - Stoploss should be at: " + "{:.3f}".format(self.buyPrice)
        case 5: # Buy: +0.2%
          stoploss_setting = 1
          message += "\n" + " - Possible Risk Buy, Ref at: {:.3f}".format(self.refPrice*1.003)
        case 6:
          profit_report = 1
          message += "\n" + " - RSI cross-down below 70"
        case 7:
          profit_report = 1
          message += "\n" + " - Histogram cross-down below 0"
        case 8:
          profit_report = 1
          message += "\n" + " - Bar price drop {:.2f}%".format(self.change[self.processIndex])
        case 9:
          profit_report = 1
          message += "\n" + " - Bar is " + self.candlestick
        case 10:
          profit_report = 1
          message += "\n" + " - Near the support/resistance zone"
        case 11:
          profit_report = 1
          message += "\n" + " - MA5 < MA10 warning hold"
        case 12:
          message += "\n" + " - MA5 >= MA10 possible to hold"
        case 13:
          message += "\n" + " - Predict MA10 < MA20 should sell"
        case 14:
          message += "\n" + " - Predict MA10 >= MA20 possible buy"
    
    if (stoploss_setting == 1):
      message += "\n" + " - Stoploss at : {:.3f} {:.2f}%".format(self.stoplossPrice, ((self.stoplossPrice/self.buyPrice)-1)*100); # Stoploss ATR
    if (profit_report == 1 and stoploss_setting == 0):
      message += "\n" + " - Profit now: {:.2f}%".format((currentData["close"]-self.buyPrice)/self.buyPrice*100) #*0.996

    for key in self.volumeProfiles:
      message += "\n" + " - Zone {:.3f}: {:.0f}T".format(key, self.volumeProfiles[key])

    message += "\n" + " - Close at : " + "{:.3f}".format(currentData["close"]) \
            +  "\n" + " - Value at MA5: " + "{:.3f}".format(self.ma5[self.processIndex]) \
            +  "\n" + " - Value at MA10: " + "{:.3f}".format(self.ma10[self.processIndex]) \
            +  "\n" + " - Value at MA20: " + "{:.3f}".format(self.ma20[self.processIndex])

    self.message = message

  def get_dist_plot(self, kx, ky):
    fig = go.Figure()
    fig.add_trace(go.Histogram(name='Vol Profile', x=self.close, y=self.volume, nbinsx=150, 
                              histfunc='sum', histnorm='probability density',
                              marker_color='#B0C4DE'))
    fig.add_trace(go.Scatter(name='KDE', x=kx, y=ky, mode='lines', marker_color='#D2691E'))
    return fig

  def caculate_volume_profiles(self):
    # https://medium.com/swlh/how-to-analyze-volume-profiles-with-python-3166bb10ff24
    # print(stock_data["close"].min(), " ", stock_data["close"].max(), " ", len(stock_data["close"]))
    kde_factor = 0.05
    num_samples = self.dataLength
    kde = stats.gaussian_kde(self.close,weights=self.volume,bw_method=kde_factor)
    xr = np.linspace(self.close.min(),self.close.max(),num_samples)
    kdy = kde(xr)

    # Find peaks
    min_prom = kdy.max() * 0.2
    peaks, peak_props = signal.find_peaks(kdy, prominence=min_prom)

    # Add peaks to dictionary
    volumePeaks = {}
    # pocIndex = -1
    for i in range(0, len(peaks)):
      volumePeaks[xr[peaks[i]]] = kdy[peaks[i]]/kdy.min()
      
    sortedVolumePeaks = dict(sorted(volumePeaks.items(), key=lambda item: item[1]))
    reverseVolumePeaks = dict(reversed(sortedVolumePeaks.items()))

    volumePeaks = {A:N for (A,N) in [x for x in reverseVolumePeaks.items()][:4]}
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

def schedule_analysis_stock():
  currentTime = datetime.datetime.now()+ TIME_UTC_DELTA
  print("------ Day:", currentTime, "------")
  stockList = read_stock_list()
  # Read nextTimeStamp from file otherwise default: START_TRADE_TIME
  nextTimeStamp = read_next_time_stamp()
  if currentTime >= (nextTimeStamp+TIME_PROTECT_DELTA):
    try:
      send_message_telegram("-------------- Day: " + currentTime.strftime("%d, %b %Y") + " --------------")
      for stockName in stockList:
        # print("Pair: " + stockName)
        stockTrade = Trade(stockName, currentTime, nextTimeStamp)
        stockTrade.analysis_stock()

      lastTimeStamp = convert_to_datetime(stockTrade.stockData.iloc[-1])
      if lastTimeStamp.weekday() == 4: # Friday and next time is Monday
        nextTimeStamp = lastTimeStamp + 3*TIME_INTERVAL_DELTA
      else:
        nextTimeStamp = lastTimeStamp + TIME_INTERVAL_DELTA
      write_next_time_stamp(nextTimeStamp)
      print ("------ EOD:", currentTime, "------")
      send_message_telegram("-------------- EOD: " + currentTime.strftime("%d, %b %Y") + " --------------")
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

    if not SERVER_MODE:
      # Program ended, turn off sys log file mode
      sys.stdout = old_stdout
      LOG_FILE.close()

  except Exception as ex:
    print("Program Exception: Is it related Telegram?", ex)
    if not SERVER_MODE:
      # Program ended, turn off sys log file mode
      sys.stdout = old_stdout
      LOG_FILE.close()

  # Windows sleep this task so using Window Task Schedule
  # scheduler.add_job(schedule_analysis_stock, 'cron', day_of_week="mon-fri", hour="15", minute="30", timezone=TIME_ZONE, misfire_grace_time=None)  # run on Monday to Friday at 15h30
  # try:
  #     scheduler.start()
  # except (KeyboardInterrupt, SystemExit):
  #     pass
  