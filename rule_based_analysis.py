"""
Step Height Detection Analysis Program by Matthew Chen, with inspiration and guidance from ToPick by Johnny Clapham et al.
Finds flight time of foot and uses Flight Time Equation to estimate step height
"""

import os
import csv
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as plticker
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import scipy.signal as signal
import numpy
from datetime import datetime, timedelta

# Directory where data and ground truth is located
data_dir = "data"
gt_dir = "gt"
# Generate graph for 1 specific patient, indexed by order of files
GRAPH = "84"
GRAPH_TITLE = "Data Subset Graph with Steps Shaded"
GRAPH_COLORS = {"Left":["blue", "indigo"], "Right":["orange", "tomato"]}
STEP_PERCENT_CUTOFF = 65 # Threshold percentile
STEP_GAP_LENGTH = 0.4
STEP_LENGTH = 0.5



def read_file(ug_data, first):
    # [1:] to skip space in the timestamp
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
    wyad[0][1] = left_scaler.fit_transform(numpy.reshape(wyad[0][1], (-1, 1)))
    right_scaler = StandardScaler()
    wyad[1][1] = right_scaler.fit_transform(numpy.reshape(wyad[1][1], (-1, 1)))

    # Take absolute value of data
    wyad[0][1] = numpy.abs(wyad[0][1])
    wyad[1][1] = numpy.abs(wyad[1][1])

    return wyad



""" diff = Difference between video start and data start, calculated from Unix times """
def read_gt(patient_id, first):
    gt_list = []
    with open(f"{gt_dir}{os.sep}{patient_id}.csv") as gt_file:
        gt_data = list(csv.reader(gt_file))
        # First row in GT file has Unix time, Video Time for start of GT video
        vid_start = float(gt_data[1][1])
        unix_start = int(gt_data[1][0])
        for i in range(2, len(gt_data)):
            # Step Height, Time (adjusted according to start)
            gt_list.append([float(gt_data[i][0]), float(gt_data[i][1]) - vid_start - (first - unix_start)/1000])
    return gt_list



def graph(wyad, thresholds=None, steps=None):
    # Plot only a portion of the data so that it is readable
    beninging = 1600
    subset = 2300
    offset = 150 # Shifts right Y accelerometer in case data is uneven

    # Finds corresponding time of these indices, assumes same length for left and right IMUs
    time_span = [wyad[0][2][beninging], wyad[0][2][subset]]

    matplotlib.rcParams.update({'font.size': 16})
    
    # Merge left and right into a single plot
    # fig, ax = plt.subplots()
    # plt.plot(wyad[0][2][beninging:subset], wyad[0][1][beninging:subset], 
    #          color=GRAPH_COLORS[wyad[0][0]][0], label=wyad[0][0] + " Y Accelerometer")
    # plt.plot(wyad[1][2][beninging - offset:subset - offset], wyad[1][1][beninging - offset:subset - offset], 
    #          color=GRAPH_COLORS[wyad[1][0]][0], label=wyad[1][0] + " Y Accelerometer")
    # ax.legend(loc="upper left")

    # Split left and right into their own subplots
    fig, ax = plt.subplots(2, sharex=True)
    ax[0].plot(wyad[0][2][beninging:subset], wyad[0][1][beninging:subset], 
               color=GRAPH_COLORS[wyad[0][0]][0], label=wyad[0][0] + " Y Accelerometer")
    ax[0].legend(loc="upper left")
    ax[1].plot(wyad[1][2][beninging - offset:subset - offset], wyad[1][1][beninging - offset:subset - offset], 
               color=GRAPH_COLORS[wyad[1][0]][0], label=wyad[1][0] + " Y Accelerometer")
    ax[1].legend(loc="upper left")

    # this locator puts ticks at regular intervals (https://stackoverflow.com/a/19972993)
    loc = plticker.MultipleLocator(base=1.0) 
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



""" Output Format: {Patient ID: [ [Left Step Timespans], [Right Foot Timespans] ]}"""
def step_analysis(ug_dict):
    step_results = {}

    for key, value in ug_dict.items():
        # Index Format: 0 Left, 1 Right
        step_results[key] = []
        thresholds = []
        for i in range(2):
            # Write CSV file of data
            # data = zip(value[i][1].flatten(), value[i][2])
            # with open("data_example.csv", "w", newline="") as f:
            #     writer = csv.writer(f)
            #     writer.writerow(["Y Accelerometer Reading", "Video Time"])
            #     writer.writerows(data)
            steps = []
            step_percent = numpy.percentile(value[i][1], STEP_PERCENT_CUTOFF)
            # Calculated Step Percentage Threshold, Left/Right
            thresholds.append([step_percent, value[i][0]])
            # Go through ug_dict and for each one, identify non-step and step portions
            step_end = 0
            step_start = -1
            in_step = False
            for j in range(len(value[i][1])):
                # this = List from specific side in WYAD
                this = value[i]

                # If ascends above threshold right after step gap, now in a step
                if not in_step and this[1][j] > step_percent:
                    if this[2][j] - step_end > STEP_GAP_LENGTH:
                        step_start = this[2][j]
                        in_step = True

                # If within a step and descends below threshold
                elif in_step and this[1][j] < step_percent:

                    # Check that part afterwards is a legit step gap
                    next_time = j+1
                    end_step = True
                    while next_time < len(this[2]) and this[2][next_time] - this[2][j] < STEP_GAP_LENGTH:
                        # Ignore tiny spikes in the middle of step gaps
                        if this[1][next_time] > step_percent and this[2][next_time] - this[2][j] < STEP_GAP_LENGTH/2:
                            end_step = False
                            break
                        next_time += 1
                    
                    # If so, record step if it's long enough and now in a step gap
                    if end_step:
                        step_end = this[2][j]
                        if step_end - step_start > STEP_LENGTH:
                            steps.append([step_start, step_end])
                        in_step = False
            step_results[key].append(steps)
        if key == GRAPH:
            # graph(value)
            # graph(value, thresholds)
            graph(value, thresholds, step_results[key])
    return step_results



def extract_step_info(step_results, step_gt):
    # Essentially, step height is x and flight time is y
    step_heights = []
    step_timespans = []

    # Compare step timespan to actual step height
    for step_time in step_results:
        start = step_time[0]
        end = step_time[1]
        gt_index = 0

        while gt_index < len(step_gt):
            # print(f"start: {start}     end: {end}")
            # print(step_gt[gt_index][1])
            if step_gt[gt_index][1] > start and step_gt[gt_index][1] < end:
                step_heights.append(step_gt[gt_index][0])
                step_timespans.append(round(end - start, 3))
                break
            gt_index += 1

    return step_heights, step_timespans



def graph_models(step_heights, step_timespans, test_heights=None, test_timespans=None, title=None):
    # Only run STE on test timespans if it's a different set than training
    test_ste_predicted_steps = []

    if test_heights == None:
        test_heights = step_heights
    if test_timespans == None:
        test_timespans = step_timespans
        test_ste_predicted_steps = None

    ste_predicted_steps = []
    for time in step_timespans:
        # Step Timespan Equation
        ste_predicted_steps.append(round((time**2 * 386.0886) / 8, 3))

    if test_ste_predicted_steps != None:
        for time in test_timespans:
            # Step Timespan Equation
            test_ste_predicted_steps.append(round((time**2 * 386.0886) / 8, 3))
    else:
        test_ste_predicted_steps = ste_predicted_steps

    linear_model = LinearRegression()
    step_timespans = numpy.array(step_timespans).reshape(-1, 1)
    linear_model.fit(step_timespans, step_heights)
    test_timespans = numpy.array(test_timespans).reshape(-1, 1)
    linear_predicted_steps = linear_model.predict(test_timespans)

    # poly = PolynomialFeatures(degree=2)
    # st_poly = poly.fit_transform(numpy.array(flight_times))
    # poly_model = LinearRegression()
    # poly_model.fit(st_poly, step_heights)
    # poly_predicted_steps = poly_model.predict(st_poly)

    ste_model = LinearRegression()
    ste_predicted_steps = numpy.array(ste_predicted_steps).reshape(-1, 1)
    ste_model.fit(ste_predicted_steps, step_heights)
    test_ste_predicted_steps = numpy.array(test_ste_predicted_steps).reshape(-1, 1)
    linear_ste_predicted_steps = ste_model.predict(test_ste_predicted_steps)

    plt.scatter(test_timespans, test_heights, color="k", label="Ground Truth")
    plt.plot(test_timespans, linear_predicted_steps, label="Linear Regression")
    plt.scatter(test_timespans, test_ste_predicted_steps, color="orange", label="Step Timespan Equation (STE)")
    plt.scatter(test_timespans, linear_ste_predicted_steps, color="red", label="STE + Linear Regression")
    plt.ylabel("Step Height")
    plt.xlabel("Step Timespan")
    plt.legend(loc="upper left")
    if title != None:
        plt.title(title)
    plt.show()

    print(f"Linear Regression MAE: {mean_absolute_error(test_heights, linear_predicted_steps)}")
    print(f"Step Timespan Equation MAE: {mean_absolute_error(test_heights, test_ste_predicted_steps)}")
    print(f"STE + Linear Regression MAE: {mean_absolute_error(test_heights, linear_ste_predicted_steps)}")


def main():
    # Dictionary of IMU data. {Patient ID: WYAD}
    ug_dict = {}
    # Step Ground Truth (not Game Theory). {Patient ID: [[Step Height], [Time]]}
    step_gt = {}

    patient_files = [x.split(".")[0] for x in os.listdir(gt_dir) if x.split(".")[-1] == "csv"]
    """ Read IMU data into ug_dict and ground truth into step_gt """
    for file in os.listdir(data_dir):
        with open(data_dir + os.sep + file) as patient_file:
            # UG Data: UltiGesture IMU data
            ug_data = list(csv.reader(patient_file))
            # This was the first; it has seen everything
            first = int(ug_data[6][1])

            # Code to only analyze a specific patient
            if ug_data[0][1] != GRAPH:

            # Code to leave out 1 patient
            # if ug_data[0][1] == "85" or ug_data[0][1] not in patient_files:
                continue

            # WYAD: Walking Y Accelerometer Data [["Left", [IMU Readings], [Time]], ["Right", [IMU Readings], [Time]]]
            wyad = read_file(ug_data, first)      
            ug_dict[ug_data[0][1]] = wyad

            # Read associated ground truth of the patient.
            step_gt[ug_data[0][1]] = read_gt(ug_data[0][1], first)

            # Generate graph of unanalyzed data
            # if GRAPH == patient_num:
            #     graph(wyad)
            # start_time = int((datetime.strptime(ug_data[5][1][1:], "%Y%m%d%H%M%S%f") - datetime(1970, 1, 1)) / timedelta(milliseconds=1))
        # break
    
    step_results = step_analysis(ug_dict)

    train_patients = []
    train_heights = []
    train_timespans = []

    test_patient = "84"
    test_heights = []
    test_timespans = []

    # colors = {"25":"red", "26":"green", "83":"blue", "84":"yellow"}
    # figure, axis = plt.subplots()
    # axis.set_ylabel("Step Height")
    # axis.set_xlabel("Step Timespan")


    # Either graph for individual patients or graph cumulatively for each patient
    for key, value in step_results.items():
        # print(step_gt[key])
        step_heights, step_timespans = extract_step_info(value[0], step_gt[key])

        # if key == test_patient:
        #     test_heights = step_heights
        #     test_timespans = step_timespans
        # else:
        #     train_heights += step_heights
        #     train_timespans += step_heights
        #     train_patients.append(key)

        graph_models(step_heights, step_timespans)

        # plt.scatter(step_timespans, step_heights, color="k", label="Ground Truth")
        # plt.title("Patient " + key)
        # plt.ylabel("Step Height")
        # plt.xlabel("Step Timespan")
        # plt.show()

        # axis.scatter(step_timespans, step_heights, color=colors[key], label=f"Patient {key}")
    
    # graph_models(train_heights, train_timespans, test_heights, test_timespans, title=f"Patient {test_patient} Left Out")
    # figure.legend(loc="center right")
    # plt.show()



if __name__ == "__main__":
    main()