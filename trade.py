import sys
import os 
import time
import pandas as pd
import datetime
import telegram
# Scheduler
# from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.schedulers.background import BackgroundScheduler
from pytz import utc,timezone
#import asyncio
import ccxt
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
parser.add_argument("-sb", "--sandbox", help="Sandbox mode to test trading BTC statistics "
                                             "by default program will run in normal mode ",
                    action="store_true", default=False)
parser.add_argument("-s", "--server-mode", help="Run program in server like pythonanywhere/heroku "
                                                "by default program will run in local machine",
                    action="store_true", default=False)
args = parser.parse_args()

SERVER_MODE = args.server_mode # Heroku Server: GMT +0, print log
SAND_BOX_MODE = args.sandbox
STOCK_FILE = DIR_PATH+"/Database/ToTheMars.tls"

if SAND_BOX_MODE:
  print ("SANDBOX MODE")
  NEXT_TIME_FILE = CACHE_PATH + "/NextTimeFile-Sandbox.txt"
  LOG_PATH = CACHE_PATH + "/trade-sandbox.log"
  START_TRADE_TIME = datetime.datetime.now() #GMT+7 Current time
  START_INDEX_TIME_DELTA = datetime.timedelta(hours = 7, minutes=120) #Zone compare & start point
  TIME_PROTECT_DELTA = datetime.timedelta(hours = 7, minutes=60, seconds = 2) # Add 2s to prevent missing data for 1h interval 
  TIME_INTERVAL_STR = '1h'
  TELEGRAM_API_ID = "5588630950:AAHXR7kcwpwdPVEvx2YwXUVeuQ_Qaz_wPNQ"
  TELEGRAM_CHANNEL_ID = "@telephaisinhtienao"
  ## Sandbox mode don't using old data file:
  if os.path.isfile(NEXT_TIME_FILE):
    os.remove(NEXT_TIME_FILE)
  TIME_ZONE = timezone("Asia/Ho_Chi_Minh")
  # Scheduler for any plans
  scheduler = BackgroundScheduler()
  # scheduler = BlockingScheduler()
  # configure exchange
  # 'apiKey': 'yI3PFfXDUgaU2VulE4N2IIosGDtLyLEtkAookD6JHWba55G8itCwXwlZk2yrreC6',
  # 'secret': 'S1k5SbLkEc6XpQJOv5VBGFBY0srujicUGt0RxpLL0wswKeBseieUlTAAFjXYGb7D',
  exchange = ccxt.binance({
    'timeout': 10000,
    'enableRateLimit': True
  })
  # exchange.setSandboxMode("True")
  exchange.load_markets()
else:
  NEXT_TIME_FILE = CACHE_PATH + "/NextTimeFile.txt"
  LOG_PATH = CACHE_PATH + "/trade.log"
  START_TRADE_TIME = datetime.datetime.strptime("2022-03-30 15:33:00",'%Y-%m-%d %H:%M:%S') #GMT+7 Trade time
  START_INDEX_TIME_DELTA = datetime.timedelta(days = 2, hours = 7) #Zone compare
  TIME_PROTECT_DELTA = datetime.timedelta(days = 1, hours = 7, seconds = 10) # Add 10s to prevent missing data for 1 day interval 
  TIME_INTERVAL_STR = '1d'
  TELEGRAM_API_ID = "5249469985:AAF6_SyVBvigBM-s3EyghW63l64_CCbaIzw"
  TELEGRAM_CHANNEL_ID = "@bottradecryptocoin"
  # configure exchange
  exchange = ccxt.binance({
  'timeout': 10000,
  'enableRateLimit': True
  })
  
if not SERVER_MODE:
  TIME_UTC_DELTA = datetime.timedelta(hours = 0)
  # Dump print to log file
  OLD_STDOUT = sys.stdout
  LOG_FILE = open(LOG_PATH,"a")
  sys.stdout = LOG_FILE
else:
  if SAND_BOX_MODE: ##TODO: Cheat: Run Sandbox in server meaning run in local with printting log
    TIME_UTC_DELTA = datetime.timedelta(hours = 0)
  else:
    TIME_UTC_DELTA = datetime.timedelta(hours = 7)

# # load markets and all coin_pairs
# coin_pairs = exchange.symbols
# # list of coin pairs which are active and use BTC as base coin
# valid_coin_pairs = []
# # load only coin_pairs which match regex and are active
# regex = '^.*/USDT'
# for coin_pair in coin_pairs:
#   if re.match(regex, coin_pair) and exchange.markets[coin_pair]['active']:
#     valid_coin_pairs.append(coin_pair)

def convert_string_to_date(dateStr):
  #return exchange.parse8601(dateStr)
  return datetime.datetime.strptime(dateStr,'%Y-%m-%dT%H:%M:%S.%fZ')

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
    stockName = stockLine.split(" ")
    stockList.append(stockName[0])
  stockFile.close()
  # print(stockList)
  return stockList

# Read/Write format: %d-%m-%Y %H:%M:%S
if not SERVER_MODE:
  def read_next_time_stamp():
    try:
      with open(NEXT_TIME_FILE, "r") as file:
          return datetime.datetime.strptime(file.read(),'%d-%m-%Y %H:%M:%S')
    except IOError:
      return START_TRADE_TIME
else:
  def read_next_time_stamp():
    try:
      nextTimeStamp = os.environ['NEXT_TIME_STAMP']
      return datetime.datetime.strptime(nextTimeStamp,'%d-%m-%Y %H:%M:%S')
    except KeyError:
      try:
        with open(NEXT_TIME_FILE, "r") as file:
            return datetime.datetime.strptime(file.read(),'%d-%m-%Y %H:%M:%S')
      except IOError:
        return START_TRADE_TIME

def write_next_time_stamp(nextTimeStamp):
  with open(NEXT_TIME_FILE, "w+") as file:
    file.write(nextTimeStamp.strftime('%d-%m-%Y %H:%M:%S'))

class Trade:
  def __init__(self, stockName, currentTime, nextTimeStamp, timeFrame='1d'):
    self.stockName = stockName
    self.timeFrame = timeFrame
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
    if SAND_BOX_MODE:
      self.balance = 0
    self.get_historical_data()
    self.caculate_start_index()
    self.get_stock_indicators()

  def get_historical_data(self):
    """Get Historical data (ohlcv) from a coin_pair
    """
    # optional: exchange.fetch_ohlcv(coin_pair, '1h', since)
    data = exchange.fetch_ohlcv(self.stockName, self.timeFrame, None, 366) #Null/None
    # update timestamp to human readable timestamp
    data = [[exchange.iso8601(candle[0])] + candle[1:] for candle in data]
    header = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    self.stockData = pd.DataFrame(data, columns=header)
    self.dataLength = len(self.stockData)

  def caculate_start_index(self):
    for index in range (1, (self.dataLength)):
      tmpTime = convert_string_to_date(self.stockData.iloc[index]["timestamp"])
      if tmpTime > (START_TRADE_TIME-START_INDEX_TIME_DELTA):
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
    print("#" + self.stockName + " Day: " , self.nextTimeStamp, "UTC+7")
    for i in range(self.startIndex, (self.dataLength-1)): # Crypto: don't use last value
      if i < 20: # Condition to fit MA20 has value
        continue
      self.commands = [] # 0: Do nothing, 1: Buy, 2: Sell, 3-4: Stoploss, 5: Increase stoploss, 6: Buy if missing, 7-11: Indicators 
      # Caculate prediction next day MA10 vs MA20
      nextMA10 = (self.ma10[i]*10 - self.close[i-9] + self.close[i])/10
      nextMA20 = (self.ma20[i]*20 - self.close[i-19] + self.close[i])/20
      # If the MA10 crosses MA20 line upward
      if self.ma10[i] > self.ma20[i] and self.ma10[i - 1] <= self.ma20[i - 1] and self.hold == 0: # and self.macdHistogram[i] > -0.005 
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

      if self.hold == 0:
        if len(self.commands) > 0:
          if self.ma5[i] >= self.ma10[i]: # Possible to hold on Sell
            self.commands.append(13)
          if nextMA10 >= nextMA20 and self.ma10[i] < self.ma20[i]: # Warning next uptrend
            self.commands.append(14)
        else: # Not remind fake dump, call buy before real signal
          if nextMA10 >= nextMA20 and self.ma10[i] < self.ma20[i]: # Warning next uptrend
            self.commands.append(15)
            refPrice = (self.ma5[i] + self.ma10[i] + self.ma20[i])/3 # Reference for buy
            if (refPrice > self.close[i]):
              self.refPrice = self.close[i]
            else:
              self.refPrice = refPrice
            self.stoplossPrice = self.refPrice - self.atr[i] # Stoploss based ATR(10)

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

        # Buy if have a chance, refPrice <-> close: 10%
        if (self.close[i]*0.9) <= self.refPrice and (len(self.commands) == 0 or self.commands[0] > 2):
          self.commands.append(5)

      if self.hold == 1 or len(self.commands) > 0:
        # Cross RSI, MCDA, MA, -8.5% change
        if self.rsi[i] < 70 and self.rsi[i - 1] >= 70:
          self.commands.append(6)

        if self.macdHistogram[i] < 0 and self.macdHistogram[i - 1] >= 0:
          self.commands.append(7)

        if self.change[i] <= -8.5:
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
        
        if not SAND_BOX_MODE:
          if len(self.volumeProfiles) > 0:
            for key in self.volumeProfiles:
              diff = abs(self.close[i]-key)/self.close[i]
              if diff < 0.05: # Compare to resistance/support with 5%
                self.commands.append(10)
                break

        if self.ma5[i] < self.ma10[i]: #Downtrend warning
          self.commands.append(11)

      if self.hold == 1 or SAND_BOX_MODE: 
        if nextMA10 < nextMA20 and self.ma10[i] >= self.ma20[i]: # Warning next downtrend
          self.commands.append(12)

      if (self.nextTimeStamp - START_INDEX_TIME_DELTA) < convert_string_to_date(self.stockData.iloc[i]["timestamp"]):
        print ("Processing: ",self.stockName, i, convert_string_to_date(self.stockData.iloc[i]["timestamp"]), "UTC")
        # Push to telegram
        if (len(self.commands) > 0):
          self.processIndex = i
          self.prepare_message()
          send_message_telegram(self.message)
        # Trade stock  
        # if SAND_BOX_MODE:
        #   if (len(self.commands) > 0) and (self.commands[0] == 1 or 
        #       self.commands[0] == 2 or self.commands[0] == 4): # Buy/Sell - Increase stoploss
        #     self.processIndex = i
        #     self.trade_stock()
      elif i == (self.dataLength-2) and (self.nextTimeStamp - START_INDEX_TIME_DELTA) \
            >= convert_string_to_date(self.stockData.iloc[i]["timestamp"]):
        # Time is comming but no stock data fitting
        raise Exception("Check the system, " + self.stockName + " does not have target data, last: " + self.stockData.iloc[i]["timestamp"])

  # Prepare message to send anywhere
  def prepare_message(self):
    currentData = self.stockData.iloc[self.processIndex]
    message = " #" + self.stockName + " Day: " + convert_string_to_date(currentData["timestamp"]).strftime("%d, %b %Y")
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
          message += "\n" + " - Possible Buy, Ref at: {:.3f}".format(self.refPrice*1.003)
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
          if not SAND_BOX_MODE:
            profit_report = 1
            message += "\n" + " - Near the support/resistance zone"
        case 11:
          profit_report = 1
          message += "\n" + " - MA5 < MA10 warning hold"
        case 12:
          message += "\n" + " - Predict MA10 < MA20 should sell"
        case 13:
          message += "\n" + " - MA5 >= MA10 possible to hold"
        case 14:
          message += "\n" + " - Predict MA10 >= MA20 possible hold"
        case 15:
          message += "\n" + " - Predict MA10 >= MA20 possible buy"
          stoploss_setting = 1
          message += "\n" + " - Possible Buy, Ref at: {:.3f}".format(self.refPrice*1.003)
    
    if (stoploss_setting == 1):
      message += "\n" + " - Stoploss at : {:.3f} {:.2f}%".format(self.stoplossPrice, ((self.stoplossPrice/self.buyPrice)-1)*100); # Stoploss ATR
    if (profit_report == 1 and stoploss_setting == 0):
      message += "\n" + " - Profit now: {:.2f}%".format((currentData["close"]-self.buyPrice)/self.buyPrice*100) #*0.996

    if not SAND_BOX_MODE:  
      for key in self.volumeProfiles:
        message += "\n" + " - Zone {:.3f}: {:.0f}T".format(key, self.volumeProfiles[key])

    message += "\n" + " - Close at : " + "{:.3f}".format(currentData["close"]) \
            +  "\n" + " - Value at MA5: " + "{:.3f}".format(self.ma5[self.processIndex]) \
            +  "\n" + " - Value at MA10: " + "{:.3f}".format(self.ma10[self.processIndex]) \
            +  "\n" + " - Value at MA20: " + "{:.3f}".format(self.ma20[self.processIndex])

    self.message = message

  def trade_stock(self):
    self.balance = self.get_balance(self.stockName)
    for command in self.commands:
      match command:
        case 1: # Buy
          # Check before buy
          try:
            # orders = exchange.fetch_open_orders(stock_name, None, None)
            orderCancel = exchange.cancel_all_orders(self.stockName)
            print ("Order Cancel:", orderCancel)
          except Exception as e:
            print("No any orders to cancel", e)
          if self.balance > 0:
            send_message_telegram("Remain " + self.stockName + " balance: " + "{:.3f}".format(self.balance))

          # Limit buy
          # Current trade market, TODO: thread process limit order
          # capital = 100 #USD
          # amount = capital/buyPrice 
          # orderBuy = exchange.create_order(stock_name, 'limit', 'buy', amount, buyPrice)
          # Market price
          params = {
              'quoteOrderQty': 100,  # 100 USDT
          }
          amount = None
          price = None
          order = exchange.create_order(self.stockName, 'market', 'buy', amount, price, params)
          print("Order Buy:", order)

          # Stoploss settings
          # time.sleep(5) 
          # self.balance = self.get_balance(self.stockName)
          # if self.balance > 0:
          #   stop_loss_params = {'stopPrice': self.stoplossPrice}
          #   order = exchange.create_order(self.stockName, 'stop_loss_limit', 'sell', self.balance, self.stoplossPrice*0.998, stop_loss_params)
          #   print("Order Stoploss:", order)
          # else:
          #   send_message_telegram("Buy " + self.stockName + " did not done")
          #   raise Exception("Buy " + self.stockName + " did not done")

        case 2: # Sell
          try:
            orderCancel = exchange.cancel_all_orders(self.stockName)
            print ("Order Cancel:", orderCancel)
          except Exception as e:
            print("No any orders to cancel", e)
          if self.balance > 0:
            order = exchange.create_order(self.stockName, 'limit', 'sell', self.balance, self.stockData.iloc[self.processIndex]["close"]*0.998)
            print("Order Sell:", order)
          else:
            send_message_telegram("Sell " + self.stockName + " did not have balance")

        # case 4: # Increase stoploss
        #   try:
        #     orderCancel = exchange.cancel_all_orders(self.stockName)
        #     print ("Order Cancel:", orderCancel)
        #   except Exception as e:
        #     print("No any orders to cancel", e)
        #   if self.balance > 0:
        #     stop_loss_params = {'stopPrice': self.stoplossPrice}
        #     order = exchange.create_order(self.stockName, 'stop_loss_limit', 'sell', self.balance, self.stoplossPrice*0.998, stop_loss_params)
        #     print("Order +Stoploss:", order)
        #   else:
        #     send_message_telegram("+Stoploss " + self.stockName + " did not have balance")

  def get_balance(self, stockName):
    stock = stockName.split("/")
    # timestamp = exchange.milliseconds()
    balance = exchange.fetch_balance()
    totalBalance = balance["total"]
    stockBalance = 0
    if totalBalance.get(stock[0], -1) != -1:
      stockBalance = totalBalance[stock[0]]
    #print(exchange.iso8601(timestamp), 'Balance: ', stockBalance)

    return stockBalance

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
    min_prom = kdy.max() * 0.1
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
  if currentTime >= nextTimeStamp:
    try:
      send_message_telegram("-------------- Day: " + currentTime.strftime("%d, %b %Y") + " --------------")
      for stockName in stockList:
        if stockName != "BTC/USDT" and SAND_BOX_MODE:
          continue
        # print("Pair: " + stockName)
        # respect rate limit
        time.sleep(exchange.rateLimit / 1000)
        stockTrade = Trade(stockName, currentTime, nextTimeStamp, TIME_INTERVAL_STR)
        stockTrade.analysis_stock()

        # if SAND_BOX_MODE:
        #   print("Balance: ",(stockTrade.get_balance(stockName))*stockTrade.stockData.iloc[-1]["close"] \
        #     + stockTrade.get_balance("USDT"), stockTrade.stockData.iloc[-1]["close"])
          
      nextTimeStamp = convert_string_to_date(stockTrade.stockData.iloc[-1]["timestamp"]) + TIME_PROTECT_DELTA
      write_next_time_stamp(nextTimeStamp)
      print ("------ EOD:", currentTime, "------")
      send_message_telegram("-------------- EOD: " + currentTime.strftime("%d, %b %Y") + " --------------")
    except Exception as ex:
      send_message_telegram("Something wrong: " + str(ex))
      print("Something wrong: ", ex)
      # scheduler.shutdown()
  else:
    print("Please run again after 7am, next time is", nextTimeStamp)

if __name__ == "__main__":
  try:
    # Run first time if needed
    schedule_analysis_stock()
    if SAND_BOX_MODE:
      # scheduler.add_job(schedule_analysis_stock, 'interval', minutes=60, timezone=TIME_ZONE) # Recommend run at: 05s of each minute
      scheduler.add_job(schedule_analysis_stock, 'cron', minute="00", second="30", timezone=TIME_ZONE)  # run on every hour at hh:00:30
      # scheduler.start()
      
      try:
          scheduler.start()
          while(True):
            time.sleep(10)
      except (KeyboardInterrupt, SystemExit):
          pass
   
    if not SERVER_MODE:
      # Program ended, turn off sys log file mode
      sys.stdout = OLD_STDOUT
      LOG_FILE.close()

  except Exception as ex:
    print("Program Exception: Is it related Telegram?", ex)
    if not SERVER_MODE:
      # Program ended, turn off sys log file mode
      sys.stdout = OLD_STDOUT
      LOG_FILE.close()

  # Windows sleep this task so using Window Task Schedule
  # if SAND_BOX_MODE:
  #   # scheduler.add_job(schedule_analysis_stock, 'interval', minutes=1, timezone=TIME_ZONE) # Recommend run at: 05s of each minute
  #   scheduler.add_job(schedule_analysis_stock, 'cron', second="5", timezone=TIME_ZONE) # Recommend run at: 05s of each minute
  # else:
  #   scheduler.add_job(schedule_analysis_stock, 'cron', hour="7", minute="00", second="30", timezone=TIME_ZONE)  # run on everyday at 7:00
  # try:
  #     scheduler.start()
  # except (KeyboardInterrupt, SystemExit):
  #     pass