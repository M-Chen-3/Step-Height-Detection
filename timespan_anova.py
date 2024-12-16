import os
import csv
import numpy
import scipy.signal as signal
from scipy.stats import f_oneway as anova
from sklearn.preprocessing import StandardScaler

# Directory where data and ground truth is located
data_dir = "data"
gt_dir = "gt"
STEP_PERCENT_CUTOFF = 65 # Threshold percentile
STEP_GAP_LENGTH = 0.4
STEP_LENGTH = 0.5



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
    wyad[0][1] = left_scaler.fit_transform(numpy.reshape(wyad[0][1], (-1, 1)))
    right_scaler = StandardScaler()
    wyad[1][1] = right_scaler.fit_transform(numpy.reshape(wyad[1][1], (-1, 1)))

    # Take absolute value of data
    wyad[0][1] = numpy.abs(wyad[0][1])
    wyad[1][1] = numpy.abs(wyad[1][1])

    return wyad



""" diff = Difference between video start and data start, calculated from Unix times """
def read_gt(patient_id, first):
    indexes = {"Left":0, "Right":1}
    gt_list = [[], []]
    with open(f"{gt_dir}{os.sep}{patient_id}.csv") as gt_file:
        gt_data = list(csv.reader(gt_file))
        # First row in GT file has Unix time, Video Time for start of GT video
        vid_start = float(gt_data[1][1])
        unix_start = int(gt_data[1][0])
        for i in range(2, len(gt_data)):
            # Step Height, Time (adjusted according to start)
            side = indexes[gt_data[i][2]]
            gt_list[side].append([float(gt_data[i][0]), round(float(gt_data[i][1]) - vid_start - (first - unix_start)/1000, 3)])
    return gt_list



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
    return step_results



def extract_step_info(step_results, step_gt):
    # Essentially, step height is x and flight time is y
    step_heights = []
    step_timespans = []

    # Compare step timespan to actual step height
    for i in range(2):
        for step_time in step_results[i]:
            start = step_time[0]
            end = step_time[1]
            gt_index = 0
            gt = step_gt[i]

            while gt_index < len(gt):
                # print(f"start: {start}     end: {end}")
                # print(gt[gt_index][1])
                if gt[gt_index][1] > start and gt[gt_index][1] < end:
                    step_heights.append(gt[gt_index][0])
                    step_timespans.append(round(end - start, 3))
                    break
                gt_index += 1

    return step_heights, step_timespans



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

            # Code to analyze all patients
            if ug_data[0][1] == "85" or ug_data[0][1] not in patient_files:
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
        # break
    
    step_results = step_analysis(ug_dict)

    all_step_heights = []
    all_step_timespans = []

    # Either graph for individual patients or graph cumulatively for each patient
    for key, value in step_results.items():
        step_heights, step_timespans = extract_step_info(value, step_gt[key])

        all_step_heights += step_heights
        all_step_timespans += step_timespans

    percentiles = [0, 25, 50, 75]
    thresholds = []

    for p in percentiles:
        thresholds.append(numpy.percentile(all_step_heights, p))

    group_0 = []
    group_25 = []
    group_50 = []
    group_75 = []

    for i in range(len(all_step_heights)):
        h = all_step_heights[i]

        if h > thresholds[3]:
            group_75.append(all_step_timespans[i])
        elif h > thresholds[2]:
            group_50.append(all_step_timespans[i])
        elif h > thresholds[1]:
            group_25.append(all_step_timespans[i])
        else:
            group_0.append(all_step_timespans[i])

    f, p_value = anova(group_0, group_25, group_50, group_75)
    print(thresholds)
    print("{:.20f}".format(p_value))

if __name__ == "__main__":
    main()