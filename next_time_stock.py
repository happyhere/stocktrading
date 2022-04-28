import sys
import os 
import time
import datetime
import requests
import json

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

NEXT_TIME_FILE = CACHE_PATH + "/NextTimeFile-Stock.txt"
START_TRADE_TIME_ORIGINAL = datetime.datetime.strptime("2022-04-01",'%Y-%m-%d') #GMT+7 Trade time count if day > 15h else day -= 1
TIME_INTERVAL_DELTA = datetime.timedelta(days = 1) # Write next time search
TIME_UTC_DELTA = datetime.timedelta(hours = 7)
TIME_PROTECT_DELTA = datetime.timedelta(hours = 15, minutes= 10) # Add 15 hours 10 minutes to prevent missing data for 1 day interval 
API_URL = "https://api.heroku.com/apps/warm-tundra-89719/dynos"
API_PAYLOAD = {
	"command": "python3 trade_stock.py -s"
}
API_HEADER = {
	"Accept": "application/vnd.heroku+json; version=3",
	"Authorization": "Bearer b725f1fc-520f-4672-8947-f7aefcd56500"
}

#Start trade cannot be Sunday, Saturday
if START_TRADE_TIME_ORIGINAL.weekday() == 5:
  print(" START_TRADE_TIME_ORIGINAL is Saturday")
  START_TRADE_TIME = START_TRADE_TIME_ORIGINAL - TIME_INTERVAL_DELTA
elif START_TRADE_TIME_ORIGINAL.weekday() == 6:
  print("START_TRADE_TIME_ORIGINAL is Sunday")
  START_TRADE_TIME = START_TRADE_TIME_ORIGINAL - 2*TIME_INTERVAL_DELTA
else:
  START_TRADE_TIME = START_TRADE_TIME_ORIGINAL

def read_next_time_stamp():
    try:
        with open(NEXT_TIME_FILE, "r") as file:
            return datetime.datetime.strptime(file.read(),'%d-%m-%Y')
    except IOError:
        return START_TRADE_TIME

def write_next_time_stamp(nextTimeStamp):
    with open(NEXT_TIME_FILE, "w+") as file:
        file.write(nextTimeStamp.strftime('%d-%m-%Y'))

def post_formdata(payload=API_PAYLOAD, url=API_URL, headers=API_HEADER, params=None):
    """Method to send request to api heroku dynos"""
    files = []
    payload = json.dumps(payload,indent=len(payload)+1)

    for _ in range(5):
        try:
            # Add delay to avoid error from too many requests per second
            time.sleep(1.1)
            response = requests.request("POST",
                                        url,
                                        headers=headers,
                                        data=payload,
                                        params=params,
                                        files=files)
            return response.json()
        except:
            continue
    return {}

def schedule_analysis_stock():
  currentTime = datetime.datetime.now()+ TIME_UTC_DELTA
  print("------ Day:", currentTime, "------")
  # Read nextTimeStamp from file otherwise default: START_TRADE_TIME
  nextTimeStamp = read_next_time_stamp()
  if currentTime >= (nextTimeStamp+TIME_PROTECT_DELTA):
    try:
      payload = API_PAYLOAD
      envPayload = {}
      envPayload["NEXT_TIME_STAMP_STOCK"] = nextTimeStamp.strftime('%d-%m-%Y')
      payload["env"] = envPayload
      response = post_formdata(payload)
      print("API response:", response)

      if currentTime.hour >= 15:
        nextTimeStamp = currentTime
      else:
        nextTimeStamp = currentTime - TIME_INTERVAL_DELTA

      if nextTimeStamp.weekday() == 4: # Friday and next time is Monday
        nextTimeStamp = nextTimeStamp + 3*TIME_INTERVAL_DELTA
      elif currentTime.weekday() == 5: # Saturday and next time is Monday
        nextTimeStamp = nextTimeStamp + 2*TIME_INTERVAL_DELTA
      else:
        nextTimeStamp = nextTimeStamp + TIME_INTERVAL_DELTA

      write_next_time_stamp(nextTimeStamp)
      print ("------ End of Day:", currentTime, "------")
    except Exception as ex:
      print("Something wrong: ", ex)
  else:
    print("Please run again after 15pm, next time is", nextTimeStamp+TIME_PROTECT_DELTA)

if __name__ == "__main__":
    # Run first time if needed
    schedule_analysis_stock()

    # Program ended, turn off sys log file mode
    sys.stdout = old_stdout
    LOG_FILE.close()