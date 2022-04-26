
import sys
import re

try:
    with open("conventional_parking_maps.osm", "r") as file:
        text = file.read()
except IOError:
    exit(0)

# print(re.findall(r'\b\d+\b', "he33llo 42 I'm a 32 string 30,35"))

for index in re.findall(r'\b\d+\b', text):
    if int(index) > 300000 and int(index) < 900000:
        print (index)

#print(''.join(filter(str.isdigit, text)))