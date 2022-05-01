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
LOG_FILE = open(CACHE_PATH + "/trade.log","a")
sys.stdout = LOG_FILE

NEXT_TIME_FILE = CACHE_PATH + "/NextTimeFile.txt"
START_TRADE_TIME = datetime.datetime.strptime("2022-03-30 00:00:00",'%Y-%m-%d %H:%M:%S')
TIME_INTERVAL_DELTA = datetime.timedelta(days = 1) # Write next time search
TIME_UTC_DELTA = datetime.timedelta(hours = 7)
TIME_PROTECT_DELTA = datetime.timedelta(days = 1, hours = 7, seconds = 10) # Add 10s to prevent missing data for 1 day interval 
API_URL = "https://api.heroku.com/apps/warm-tundra-89719/dynos"
API_PAYLOAD = {
	"command": "python3 trade.py -s"
}
API_HEADER = {
	"Accept": "application/vnd.heroku+json; version=3",
	"Authorization": "Bearer b725f1fc-520f-4672-8947-f7aefcd56500"
}

def read_next_time_stamp():
  try:
    with open(NEXT_TIME_FILE, "r") as file:
        return datetime.datetime.strptime(file.read(),'%d-%m-%Y %H:%M:%S')
  except IOError:
    return START_TRADE_TIME

def write_next_time_stamp(nextTimeStamp):
  with open(NEXT_TIME_FILE, "w+") as file:
    file.write(nextTimeStamp.strftime('%d-%m-%Y %H:%M:%S'))

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
  if currentTime >= nextTimeStamp:
    try:
      payload = API_PAYLOAD
      envPayload = {}
      envPayload["NEXT_TIME_STAMP"] = nextTimeStamp.strftime('%d-%m-%Y %H:%M:%S')
      payload["env"] = envPayload
      response = post_formdata(payload)
      print("API response:", response)

      if currentTime.hour >= 7:
        nextTimeStamp = currentTime.replace(hour=0, minute=0, second=0, microsecond=0) + TIME_PROTECT_DELTA
      else:
        nextTimeStamp = currentTime.replace(hour=0, minute=0, second=0, microsecond=0) + TIME_PROTECT_DELTA - TIME_INTERVAL_DELTA

      write_next_time_stamp(nextTimeStamp)
      print ("----- End of Day:", currentTime, "-----")
    except Exception as ex:
      print("Something wrong: ", ex)
  else:
    print("Please run again after 15pm, next time is", nextTimeStamp)

if __name__ == "__main__":
    # Run first time if needed
    schedule_analysis_stock()

    # Program ended, turn off sys log file mode
    sys.stdout = old_stdout
    LOG_FILE.close()