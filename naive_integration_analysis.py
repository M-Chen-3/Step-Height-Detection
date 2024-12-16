"""
Step Height Detection Analysis Program by Matthew Chen, with inspiration and guidance from ToPick by Johnny Clapham et al.
Naively attempts to double integrate the entirety of a patient's data. Done for demonstration and learning purposes
"""

import os
import csv
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as plticker
import scipy.signal as signal
from scipy.integrate import cumulative_trapezoid as integral_trap
from sklearn.preprocessing import StandardScaler
import numpy
from datetime import datetime, timedelta

# Directory where data and ground truth is located
data_dir = "data"
gt_dir = "gt"
# Generate graph for 1 specific patient, indexed by order of files
GRAPH = "25"
GRAPH_TITLE = "Shaded Data Subset Graph"
GRAPH_COLORS = {"Left":["blue", "indigo"], "Right":["orange", "tomato"]}
STEP_PERCENT_CUTOFF = 65
STEP_GAP_LENGTH = 0.4
STEP_LENGTH = 0.2

def read_file(ug_data, first):
    # WYAD: Walking Y Accelerometer Data [["Left", [IMU Readings], [Time]], ["Right", [IMU Readings], [Time]]]
    wyad = {ug_data[1][1][1:]:["Left", [], []], ug_data[2][1][1:]:["Right", [], []]}

    # Go through patient's data and get left and right Y accel data
    for reading in ug_data[6:]:
        if reading[0] in wyad.keys():
            wyad[reading[0]][1].append(float(reading[3]))
            # Adjust Unix time to seconds, starting from 0
            wyad[reading[0]][2].append((int(reading[1]) - first)/1000)
    wyad = list(wyad.values())

    # Print first 10 elements of each accel
    # print(f"wyad:") 
    # for key in wyad.keys():
    #     print([wyad[key][0]] + [wyad[key][x][:10] for x in [1,2]])

    # Butterworth filter data to smooth the curves
    sus = signal.butter(5, 10, 'lp', fs=100, output='sos')
    wyad[0][1] = signal.sosfilt(sus, wyad[0][1])
    wyad[1][1] = signal.sosfilt(sus, wyad[1][1])

    # Standardize data to average of 0
    left_scaler = StandardScaler()
    wyad[0][1] = left_scaler.fit_transform(numpy.reshape(wyad[0][1], (-1, 1))).flatten()
    right_scaler = StandardScaler()
    wyad[1][1] = right_scaler.fit_transform(numpy.reshape(wyad[1][1], (-1, 1))).flatten()

    # Take absolute value of data
    # wyad[0][1] = numpy.abs(wyad[0][1])
    # wyad[1][1] = numpy.abs(wyad[1][1])

    return wyad


""" diff = Difference between video start and data start, calculated from Unix times """
def read_gt(patient_id, first):
    gt_list = []
    with open(f"{gt_dir}{os.sep}{patient_id}.csv") as gt_file:
        gt_data = list(csv.reader(gt_file))
        # First row in GT file has Unix time, Video Time for start of GT video
        vid_start = int(gt_data[1][1])
        unix_start = int(gt_data[1][0])
        for i in range(2, len(gt_data)):
            # Step Height, Time (adjusted according to start)
            gt_list.append([float(gt_data[i][0]), float(gt_data[i][1]) - vid_start - (first - unix_start)/1000])
    return gt_list



def graph(wyad, thresholds=None, steps=None):
    # Plot only a portion of the data so that it is readable
    beninging = 500
    subset = 2000
    # Finds corresponding time of these indices, assumes same length for left and right IMUs
    time_span = [wyad[0][2][beninging], wyad[0][2][subset]]

    matplotlib.rcParams.update({'font.size': 16})
    
    # Merge left and right into a single plot
    # fig, ax = plt.subplots()
    # plt.plot(wyad[0][2][beninging:subset], wyad[0][1][beninging:subset], color=GRAPH_COLORS[wyad[0][0]][0], label=wyad[0][0] + " Accelerometer")
    # plt.plot(wyad[1][2][beninging:subset], wyad[1][1][beninging:subset], color=GRAPH_COLORS[wyad[1][0]][0], label=wyad[1][0] + " Accelerometer")
    # ax.legend(loc="upper left")

    # Split left and right into their own subplots
    fig, ax = plt.subplots(2, sharex=True)
    ax[0].plot(wyad[0][2][beninging:subset], wyad[0][1][beninging:subset], color=GRAPH_COLORS[wyad[0][0]][0], label=wyad[0][0] + " Accelerometer")
    ax[0].legend(loc="upper left")
    ax[1].plot(wyad[1][2][beninging:subset], wyad[1][1][beninging:subset], color=GRAPH_COLORS[wyad[1][0]][0], label=wyad[1][0] + " Accelerometer")
    ax[1].legend(loc="upper left")
    loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals (https://stackoverflow.com/a/19972993)
    ax[0].yaxis.set_major_locator(loc)
    ax[1].yaxis.set_major_locator(loc)

    fig.set_size_inches(18, 6)
    fig.tight_layout()

    # Plot dashed line showing step threshold
    if thresholds != None:
        # print(thresholds)
        for i in range(len(thresholds)):
            ax[i].axhline(y=thresholds[i][0], color=GRAPH_COLORS[thresholds[i][1]][1], linestyle="dashed")

    # Shade step portions
    if steps != None:
        for i in range(len(steps)):
            for step in steps[i]:
                if step[0] > time_span[1]:
                    break
                elif step[0] > time_span[0]:
                    ax[i].axvspan(step[0], step[1], color=GRAPH_COLORS[thresholds[i][1]][0], alpha=0.33)

    plt.suptitle(GRAPH_TITLE, y=0.95, fontsize=30)
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s²)")
    fig.tight_layout()
    plt.show()

    # Old code for adjusting axis ticks
    # loc = plticker.MultipleLocator(base=5.0)
    # ax.yaxis.set_major_locator(loc)



def main():
    # Dictionary of IMU data. {Patient ID: WYAD}
    ug_dict = {}
    # Step Ground Truth (not Game Theory). {Patient ID: [[Step Height, Time]]}
    step_gt = {}

    """ Read IMU data into ug_dict and ground truth into step_gt """
    for file in os.listdir(data_dir):
        with open(data_dir + os.sep + file) as patient_file:
            # UG Data: UltiGesture IMU data
            ug_data = list(csv.reader(patient_file))
            # This was the first; it has seen everything
            first = int(ug_data[6][1])

            # Test code to only analyze a specific patient
            if ug_data[0][1] != GRAPH:
                continue

            # WYAD: Walking Y Accelerometer Data [["Left", [IMU Readings], [Time]], ["Right", [IMU Readings], [Time]]]
            wyad = read_file(ug_data, first)      
            ug_dict[ug_data[0][1]] = wyad

            # Read associated ground truth of the patient. [1:] to skip space in the timestamp
            step_gt[ug_data[0][1]] = read_gt(ug_data[0][1], first)

            # Generate graph of unanalyzed data
            # if GRAPH == patient_num:
            #     graph(wyad)
            # start_time = int((datetime.strptime(ug_data[5][1][1:], "%Y%m%d%H%M%S%f") - datetime(1970, 1, 1)) / timedelta(milliseconds=1))
        break

    """ Perform step analysis """
    for key, value in ug_dict.items():
        colors = ["blue", "orange"]
        # Index Format: 0 Left, 1 Right
        for i in range(2):
            # Time from specific side in WYAD
            x = value[i][2]

            # IMU data from specific side in WYAD
            acceleration = value[i][1]
            velocity = integral_trap(acceleration, x, initial=0)
            height = integral_trap(velocity, x, initial=0)
            plt.plot(x, height, color=colors[i])
            plt.show()

if __name__ == "__main__":
    main()