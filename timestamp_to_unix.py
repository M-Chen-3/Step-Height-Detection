import os
import csv
from datetime import datetime, timedelta, timezone
data_dir = "data"

timestamp = input("Enter timestamp: ")

# Convert input timestamp to EDT and ensure Unix epoch is in UTC
start_time = int((datetime.strptime(timestamp, "%Y%m%d%H%M%S%f").replace(tzinfo=timezone(-timedelta(hours=4))) - datetime(1970, 1, 1).replace(tzinfo=timezone.utc)) / timedelta(milliseconds=1))

print(start_time)

# Unused code to find corresponding patient file to timestamp
# for file in os.listdir(data_dir):
#     with open(data_dir + os.sep + file) as patient_file:
#         ug_data = list(csv.reader(patient_file))
#         # print(ug_data[5][1])
#         if ug_data[5][1] == " " + timestamp:
#             print(f"Match: {ug_data[0][1]}")