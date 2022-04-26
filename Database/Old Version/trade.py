import sys
import os 
import time
import pandas as pd
import datetime
import telegram

# Scheduler
# from apscheduler.schedulers.blocking import BlockingScheduler
# from pytz import utc,timezone

#import asyncio
import ccxt

# TA-lib: https://pypi.org/project/TA-Lib/
from talib import abstract

# import plotly.express as px
import plotly.graph_objects as go
from scipy import stats, signal
import numpy as np

dirPath = os.path.dirname(os.path.realpath(__file__))
cachePath = 'cache'
# Check whether the specified path exists or not
isExist = os.path.exists(cachePath)
if not isExist:
  # Create a new directory because it does not exist 
  os.makedirs(cachePath)

# Dump print to log file
old_stdout = sys.stdout
LOG_FILE = open(cachePath + "/trade.log","a")
sys.stdout = LOG_FILE

TELEGRAM_API_ID = "5249469985:AAF6_SyVBvigBM-s3EyghW63l64_CCbaIzw"
TELEGRAM_CHANNEL_ID = "@bottradecryptocoin"
# TIME_ZONE = timezone("Asia/Ho_Chi_Minh")
STOCK_FILE = "Database/ToTheMars.tls"
SAND_BOX_MODE = 0 #Simulation: 0: OFF - Bugs: Schedule near the 01~00~59, missing frame

if SAND_BOX_MODE == 1:
  NEXT_TIME_FILE = cachePath + "/NextTimeFile-Sandbox.txt"
  START_TRADE_TIME = datetime.datetime.now() #GMT+7 Current time
  START_INDEX_TIME_DELTA = datetime.timedelta(hours = 7, seconds=120) #Zone compare & start point
  TIME_PROTECT_DELTA = datetime.timedelta(hours = 7, seconds = 62) # Add 2s to prevent missing data for 1m interval 
  TIME_INTERVAL_STR = '1m'
  ## Sandbox mode don't using old data file:
  if os.path.isfile(NEXT_TIME_FILE):
    os.remove(NEXT_TIME_FILE)
else:
  NEXT_TIME_FILE = cachePath + "/NextTimeFile.txt"
  START_TRADE_TIME = datetime.datetime.strptime("2022-03-30 15:33:00",'%Y-%m-%d %H:%M:%S') #GMT+7 Trade time
  START_INDEX_TIME_DELTA = datetime.timedelta(days = 2, hours = 7) #Zone compare
  TIME_PROTECT_DELTA = datetime.timedelta(days = 1, hours = 7, seconds = 10) # Add 10s to prevent missing data for 1 day interval 
  TIME_INTERVAL_STR = '1d'

# # Scheduler for any plans
# scheduler = BlockingScheduler()

# configure exchange
if (SAND_BOX_MODE == 1):
  print ("SANDBOX MODE")
  exchange = ccxt.binance({
    'apiKey': 'yI3PFfXDUgaU2VulE4N2IIosGDtLyLEtkAookD6JHWba55G8itCwXwlZk2yrreC6',
    'secret': 'S1k5SbLkEc6XpQJOv5VBGFBY0srujicUGt0RxpLL0wswKeBseieUlTAAFjXYGb7D',
    'timeout': 10000,
    'enableRateLimit': True
  })
  exchange.setSandboxMode("True")
  exchange.load_markets()
else:
  exchange = ccxt.binance({
  'timeout': 10000,
  'enableRateLimit': True
  })


# # load markets and all coin_pairs
# coin_pairs = exchange.symbols
# # list of coin pairs which are active and use BTC as base coin
# valid_coin_pairs = []
# # load only coin_pairs which match regex and are active
# regex = '^.*/USDT'
# for coin_pair in coin_pairs:
#   if re.match(regex, coin_pair) and exchange.markets[coin_pair]['active']:
#     valid_coin_pairs.append(coin_pair)

def get_historical_data(coin_pair, timeframe):
  """Get Historical data (ohlcv) from a coin_pair
  """
  # optional: exchange.fetch_ohlcv(coin_pair, '1h', since)
  data = exchange.fetch_ohlcv(coin_pair, timeframe, None, 366) #Null/None
  # update timestamp to human readable timestamp
  data = [[exchange.iso8601(candle[0])] + candle[1:] for candle in data]
  header = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
  df = pd.DataFrame(data, columns=header)
  return df

def convert_string_to_date(dateStr):
  #return exchange.parse8601(dateStr)
  return datetime.datetime.strptime(dateStr,'%Y-%m-%dT%H:%M:%S.%fZ')

def caculate_start_index(stock_data):
  for index in range (1, (len(stock_data))):
    tmpTime = convert_string_to_date(stock_data.iloc[index]["timestamp"])
    if tmpTime > (START_TRADE_TIME-START_INDEX_TIME_DELTA):
      return index

  return len(stock_data) # Mean don't analysis anything

def analysis_stock(stock_name, stock_data, nextTimeStamp):
  """ Analysis one stock
  """
  # Caculate indicator
  volumeProfiles = caculate_volume_profiles(stock_data)
  stock_data["ma5"] = ma5 = abstract.SMA(stock_data, timeperiod=5)
  stock_data["ma10"] = ma10 = abstract.SMA(stock_data, timeperiod=10)
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
  print("#" + stock_name + " Day: " , nextTimeStamp, "UTC+7")
  for i in range(index, (len(stock_data)-1)): # Crypto: don't use last value
      commands = [] # 0: Do nothing, 1: Buy, 2: Sell, 3-4: Stoploss, 5: Increase stoploss, 6: Buy if missing, 7-11: Indicators 
      # If the MA5 crosses MA10 line upward
      if ma5[i] > ma10[i] and ma5[i - 1] <= ma10[i - 1] and macd_histogram[i] > -0.005:
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
      if ma5[i] < ma10[i] and ma5[i - 1] >= ma10[i - 1] and hold == 1 and len(commands) == 0:
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

      if change[i] <= -8.5 and hold == 1 and (len(commands) == 0 or commands[0] > 3):
        commands.append(8)

      if invertHammer[i] != 0 and hold == 1 and (len(commands) == 0 or commands[0] > 3):
        commands.append(9)
      
      if SAND_BOX_MODE == 0:
        if len(volumeProfiles) > 0 and hold == 1 and (len(commands) == 0 or commands[0] > 3):
          for key in volumeProfiles:
            diff = abs(close[i]-key)/close[i]
            if diff < 0.05: # Compare to resistance/support with 5%
              commands.append(10)
              break

      if (nextTimeStamp - START_INDEX_TIME_DELTA) < convert_string_to_date(stock_data.iloc[i]["timestamp"]):
        print ("Processing: ",stock_name, i, convert_string_to_date(stock_data.iloc[i]["timestamp"]), "UTC")
        # Push to telegram
        if (len(commands) > 0):
          message = prepare_message(stock_name, commands, stock_data.iloc[i], buyPrice, volumeProfiles, stoplossPrice)
          send_message_telegram(message)
        # Trade stock  
        if SAND_BOX_MODE == 1:
          if (len(commands) > 0) and (commands[0] == 1 or commands[0] == 2 or commands[0] == 4): # Buy/Sell - Increase stoploss
            trade_stock(stock_name, commands, stock_data.iloc[i], buyPrice, stoplossPrice)
      elif i == (len(stock_data)-2) and (nextTimeStamp - START_INDEX_TIME_DELTA) >= convert_string_to_date(stock_data.iloc[i]["timestamp"]):
        # Time is comming but no stock data fitting
        raise Exception("Check the system, " + stock_name + " does not have next data: " + stock_data.iloc[i]["timestamp"])

# Prepare message to send anywhere
def prepare_message(stock_name, commands, stock_data, buyPrice, volumeProfiles, stoplossPrice):
  message = " #" + stock_name + " Day: " + convert_string_to_date(stock_data["timestamp"]).strftime("%d, %b %Y")
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
        message += "\n" + " - Possible Buy, Old at: {:.3f}".format(buyPrice)
      case 6:
        profit_report = 1
        message += "\n" + " - RSI cross-down below 70"
      case 7:
        profit_report = 1
        message += "\n" + " - Histogram cross-down below 0"
      case 8:
        profit_report = 1
        message += "\n" + " - Bar price drop-down below 8.5%"
      case 9:
        profit_report = 1
        message += "\n" + " - Bar is invert-Hammer check balance"
      case 10:
        if SAND_BOX_MODE == 0:
          zone_setting = 1
          message += "\n" + " - Near the support/resistance zone"
  
  if (stoploss_setting == 1):
    message += "\n" + " - Stoploss at : {:.3f} {:.2f}%".format(stoplossPrice, ((stoplossPrice/buyPrice)-1)*100); # Stoploss > 8.5%
  if (profit_report == 1 and stoploss_setting == 0):
    message += "\n" + " - Profit at: {:.2f}%".format((stock_data["close"]-buyPrice)/buyPrice*100) #*0.996
  if SAND_BOX_MODE == 0:  
    if (zone_setting == 1):
      for key in volumeProfiles:
        message += "\n" + " - Zone {:.3f}: {:.2f}M".format(key, volumeProfiles[key])

  message += "\n" + " - Close at : " + "{:.3f}".format(stock_data["close"]) \
			    +  "\n" + " - Value at MA5: " + "{:.3f}".format(stock_data['ma5']) \
			    +  "\n" + " - Value at MA10: " + "{:.3f}".format(stock_data['ma10'])

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

def trade_stock(stock_name, commands, stock_data, buyPrice, stoplossPrice):
  stockBalance = get_balance(stock_name)

  for command in commands:
    match command:
      case 1: # Buy
        # Check before buy
        try:
          # orders = exchange.fetch_open_orders(stock_name, None, None)
          orderCancel = exchange.cancel_all_orders(stock_name)
          print ("Order Cancel:", orderCancel)
        except Exception as e:
          print("No any orders to cancel", e)
        if stockBalance > 0:
          send_message_telegram("Remain " + stock_name + " balance: " + "{:.3f}".format(stockBalance))

        # Market price
        params = {
            'quoteOrderQty': 100,  # 100 USDT
        }
        amount = None
        price = None
        order = exchange.create_order(stock_name, 'market', 'buy', amount, price, params)
        # Limit buy
        # Current trade market, TODO: thread process limit order
        # capital = 100 #USD
        # amount = capital/buyPrice 
        # orderBuy = exchange.create_order(stock_name, 'limit', 'buy', amount, buyPrice)
        print("Order Buy:", order)
        time.sleep(5) 
        stockBalance = get_balance(stock_name)

        if stockBalance > 0:
          stockBalance = (get_balance(stock_name))
          stop_loss_params = {'stopPrice': stoplossPrice}
          order = exchange.create_order(stock_name, 'stop_loss_limit', 'sell', stockBalance, stoplossPrice*0.998, stop_loss_params)
          print("Order Stoploss:", order)
        else:
          send_message_telegram("Buy " + stock_name + " did not done")
          raise Exception("Buy " + stock_name + " did not done")

      case 2: # Sell
        try:
          orderCancel = exchange.cancel_all_orders(stock_name)
          print ("Order Cancel:", orderCancel)
        except Exception as e:
          print("No any orders to cancel", e)
        if stockBalance > 0:
          order = exchange.create_order(stock_name, 'limit', 'sell', stockBalance, stock_data["close"]*0.998)
          print("Order Sell:", order)
        else:
          send_message_telegram("Sell " + stock_name + " did not have balance")

      case 4: # Increase stoploss
        try:
          orderCancel = exchange.cancel_all_orders(stock_name)
          print ("Order Cancel:", orderCancel)
        except Exception as e:
          print("No any orders to cancel", e)
        if stockBalance > 0:
          stop_loss_params = {'stopPrice': stoplossPrice}
          order = exchange.create_order(stock_name, 'stop_loss_limit', 'sell', stockBalance, stoplossPrice*0.998, stop_loss_params)
          print("Order +Stoploss:", order)
        else:
          send_message_telegram("+Stoploss " + stock_name + " did not have balance")

def get_balance(stock_name):
  stock = stock_name.split("/")
  # timestamp = exchange.milliseconds()
  balance = exchange.fetch_balance()
  totalBalance = balance["total"]
  stockBalance = 0
  if totalBalance.get(stock[0], -1) != -1:
    stockBalance = totalBalance[stock[0]]
  #print(exchange.iso8601(timestamp), 'Balance: ', stockBalance)

  return stockBalance

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
    min_prom = kdy.max() * 0.1
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
    stockName = stockLine.split(" ")
    stockList.append(stockName[0])
  stockFile.close()
  # print(stockList)
  
  return stockList

def read_next_time_stamp():
  try:
    with open(NEXT_TIME_FILE, "r") as file:
        return datetime.datetime.strptime(file.read(),'%d-%m-%Y %H:%M:%S')
  except IOError:
    return START_TRADE_TIME

def write_next_time_stamp(nextTimeStamp):
  with open(NEXT_TIME_FILE, "w+") as file:
    file.write(nextTimeStamp.strftime('%d-%m-%Y %H:%M:%S'))

def schedule_analysis_stock():
  currentTime = datetime.datetime.now()
  print("------ Day:", currentTime, "------")
  stockList = read_stock_list()
  # Read nextTimeStamp from file otherwise default: START_TRADE_TIME
  nextTimeStamp = read_next_time_stamp()
  if currentTime >= nextTimeStamp:
    try:
      send_message_telegram("-------------- Day: " + currentTime.strftime("%d, %b %Y") + " --------------")
      for stockName in stockList:
        if stockName != "BTC/USDT" and SAND_BOX_MODE == 1:
          continue
        # print("Pair: " + stockName)
        # respect rate limit
        time.sleep (exchange.rateLimit / 1000)
        stock_data = get_historical_data(stockName, TIME_INTERVAL_STR)
        analysis_stock(stockName, stock_data, nextTimeStamp)

        if SAND_BOX_MODE == 1:
          print("Balance: ",(get_balance(stockName))*stock_data.iloc[-1]["close"]+get_balance("USDT"), stock_data.iloc[-1]["close"])
          
      nextTimeStamp = convert_string_to_date(stock_data.iloc[-1]["timestamp"]) + TIME_PROTECT_DELTA
      write_next_time_stamp(nextTimeStamp)
      print ("------ End of Day:", currentTime, "------")
      send_message_telegram("---------- End of Day: " + currentTime.strftime("%d, %b %Y") + " ----------")
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

    # Program ended, turn off sys log file mode
    sys.stdout = old_stdout
    LOG_FILE.close()

  except Exception as ex:
    print("Program Exception: Is it related Telegram?", ex)
    # Program ended, turn off sys log file mode
    sys.stdout = old_stdout
    LOG_FILE.close()

  # Windows sleep this task so using Window Task Schedule
  # if SAND_BOX_MODE == 1:
  #   # scheduler.add_job(schedule_analysis_stock, 'interval', minutes=1, timezone=TIME_ZONE) # Recommend run at: 05s of each minute
  #   scheduler.add_job(schedule_analysis_stock, 'cron', second="5", timezone=TIME_ZONE) # Recommend run at: 05s of each minute
  # else:
  #   scheduler.add_job(schedule_analysis_stock, 'cron', hour="7", minute="00", second="30", timezone=TIME_ZONE)  # run on everyday at 7:00
  # try:
  #     scheduler.start()
  # except (KeyboardInterrupt, SystemExit):
  #     pass