"""
Step Height Detection Analysis Program by Matthew Chen, with inspiration and guidance from ToPick by Johnny Clapham et al.
Uses rotation matrices to adjust IMU orientation to fit with gravity and integrates 2nd peak
"""

import os
import csv
import copy
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as plticker
import scipy.signal as signal
from scipy.signal import find_peaks
from scipy.integrate import cumulative_trapezoid as integral
from scipy.spatial.transform import Rotation as R
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import numpy as np

# Directory where data and ground truth is located
data_dir = "data"
gt_dir = "gt"
# Generate graph for 1 specific patient, indexed by order of files
GRAPH = "25"
GRAPH_TITLE = "Shaded Data Subset Graph"
GRAPH_COLORS = {"Left":["blue", "indigo"], "Right":["orange", "tomato"]}
STEP_GAP_LENGTH = 0.25

# From https://stackoverflow.com/a/59204638/13736952
def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def read_file(ug_data, first):
    # [1:] to skip space in the timestamp
    # WYAD: Walking Y Accelerometer Data [["Left", [IMU Readings], [Time], [Angles]], ["Right", [IMU Readings], [Time], [Angles]]]
    wyad = {ug_data[1][1][1:]:["Left", [], [], []], ug_data[2][1][1:]:["Right", [], [], []]}

    # Go through patient's data and get left and right Y accel data
    # Columns are UG ID, Unix Time (ms), Accelerometer X, Y, Z, Gyroscope X, Y, Z, Magnetometer X, Y, Z, Beacon location (always null)
    for reading in ug_data[6:]:
        if reading[0] in wyad.keys():
            wyad[reading[0]][1].append(float(reading[3]))
            # Adjust Unix time to seconds, starting from 0
            wyad[reading[0]][2].append((int(reading[1]) - first)/1000)
            wyad[reading[0]][3].append([float(reading[2]), float(reading[4])])
    wyad = list(wyad.values())

    # Print first 10 elements of each accel
    # print(f"wyad:") 
    # for key in wyad.keys():
    #     print([wyad[key][0]] + [wyad[key][x][:10] for x in [1,2]])

    # Butterworth filter data to smooth the curves
    sus = signal.butter(5, 10, 'lp', fs=100, output='sos')
    wyad[0][1] = signal.sosfilt(sus, wyad[0][1])
    wyad[1][1] = signal.sosfilt(sus, wyad[1][1])

    # Non-absoluted data
    non_ab_wyad = copy.deepcopy(wyad)

    # Standardize data to average of 0
    left_scaler = StandardScaler()
    wyad[0][1] = left_scaler.fit_transform(np.reshape(wyad[0][1], (-1, 1)))
    right_scaler = StandardScaler()
    wyad[1][1] = right_scaler.fit_transform(np.reshape(wyad[1][1], (-1, 1)))

    # Take absolute value of data
    wyad[0][1] = np.abs(wyad[0][1])
    wyad[1][1] = np.abs(wyad[1][1])

    return wyad, non_ab_wyad



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
    beninging = 500
    subset = len(wyad[0][2]) - 1
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

    plt.suptitle(GRAPH_TITLE, y=0.95, fontsize=30)
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s²)")
    fig.tight_layout()
    plt.show()

    # Old code for adjusting axis ticks
    # loc = plticker.MultipleLocator(base=5.0)
    # ax.yaxis.set_major_locator(loc)



def hill_analysis(ug_dict, subset=-1):
    # Test code for graphing non-absolute data
    global non_ab_ug_dict

    # Patient ID: [ [Left Steps], [Right Steps] ] where each Step = [Start Trough, Peak, End Trough]
    all_steps = {}

    for key, value in ug_dict.items():
        # Iterate through both left and right data
        this_steps = []
        for i in range(2):
            imu_data = value[i][1].flatten()
            time_data = np.array(value[i][2])
            min_peak_height = 0.75
            if subset == -1:
                subset = [0, len(time_data)-1]

            # Get peaks from IMU data above the threshold
            step_indices = find_peaks(imu_data, height=min_peak_height, width=0.1)
            peaks, _ = step_indices
            # Original index of each peak, use for Y-axis
            peaks = [p for p in peaks if p >= subset[0] and p <= subset[1]]

            # Get troughs from reversed IMU data below a threshold
            flipped_imu_data = np.array([-i for i in imu_data])
            troughs, _ = find_peaks(flipped_imu_data, height=-min_peak_height, width=0.1)
            # Original index of each peak, use for Y-axis
            troughs = [t for t in troughs if t >= subset[0] and t <= subset[1]]

            # Plot peaks and troughs
            for x in range(1):
                break

                # Indexes adjusted accordingly for the graph, use for X-axis
                plt.plot(time_data[subset[0] : subset[1]], imu_data[subset[0] : subset[1]], color=GRAPH_COLORS[value[i][0]][0])
                plt.plot(time_data[peaks], imu_data[peaks], "x", color="Red")
                plt.hlines(min_peak_height, xmin=time_data[subset[0]], xmax=time_data[subset[1]], color=GRAPH_COLORS[value[i][0]][1], linestyle="dashed")
                direct = "Left" if i == 0 else "Right"
                plt.xlabel("Time (s)")
                plt.ylabel("Acceleration (m/s²)")
                plt.title(f"{direct} Data Peaks for Patient {GRAPH}")
                plt.show()
                
                # Indexes adjusted accordingly for the graph, use for X-axis
                adjusted_troughs = [t - subset[0] for t in troughs]
                plt.plot(time_data[subset[0] : subset[1]], flipped_imu_data[subset[0] : subset[1]], color=GRAPH_COLORS[value[i][0]][0])
                plt.plot(time_data[troughs], flipped_imu_data[troughs], "x", color="Green")
                plt.hlines(-min_peak_height, xmin=time_data[subset[0]], xmax=time_data[subset[1]], color=GRAPH_COLORS[value[i][0]][1], linestyle="dashed")
                direct = "Left" if i == 0 else "Right"
                plt.xlabel("Time (s)")
                plt.ylabel("Acceleration (m/s²)")
                plt.title(f"{direct} Data Troughs for Patient {GRAPH}")
                plt.show()
            
            # Left Trough, Peak, Right Trough
            hills = []
            global_peaks = []
            rights = []
            lefts = []
            sames = []
            # Iterate through each peak and find neighboring troughs
            cur_t = 0
            for peak in peaks:
                hill = [0, peak, 0]
                # Go forward until trough is directly past the peak
                while troughs[cur_t] < peak:
                    cur_t += 1
                hill[2] = troughs[cur_t]
                hill[0] = troughs[cur_t - 1]

                # If surrounding troughs of current peak are exactly the same as the previous one
                if len(rights) != 0 and (hill[2] == rights[-1] and hill[0] == lefts[-1]):
                    sames.append(hill)
                # If different and there are previous same hills
                elif len(sames) > 0:
                    sames.insert(0, hills[-1])
                    merged_peaks = [s[1] for s in sames] +  [hill[1]]
                    # Create new hill with same troughs and max of the sames's peaks
                    merged_hill = [sames[0][0], max(([m, imu_data[m]] for m in merged_peaks), key=lambda x: x[1])[0], sames[0][2]]
                    hills[-1] = merged_hill
                    global_peaks[-1] = merged_hill[1]
                    sames = []
                
                # Append current hill to the hills list
                if len(sames) == 0:
                    hills.append(hill)
                    global_peaks.append(hill[1])
                    rights.append(hill[2])
                    lefts.append(hill[0])
            
            # Add last hill and merge sames just in case
            sames.insert(0, hills[-1])
            merged_peaks = [s[1] for s in sames] +  [hill[1]]
            merged_hill = [sames[0][0], max(([m, imu_data[m]] for m in merged_peaks), key=lambda x: x[1])[0], sames[0][2]]
            hills[-1] = merged_hill
            global_peaks[-1] = merged_hill[1]

            # Separate hills into their respective steps using the step gap length
            steps = []
            prev_hill = hills[0]
            step = []
            for hill in hills:
                if time_data[hill[0]] - time_data[prev_hill[2]] > STEP_GAP_LENGTH:
                    steps.append(step)
                    step = [hill]
                else:
                    step.append(hill)
                prev_hill = hill
            steps.append(step)
            this_steps.append(steps)

            # Test code for graphing non-absolute data
            non_ab_imu_data = non_ab_ug_dict[key][i][1]
            plt.plot(time_data[subset[0] : subset[1]], non_ab_imu_data[subset[0] : subset[1]], color=GRAPH_COLORS[value[i][0]][0])
            # plt.plot(time_data[global_peaks], non_ab_imu_data[global_peaks], "x", color="Red")
            # plt.plot(time_data[troughs], non_ab_imu_data[troughs], "x", color="Green")
            
            # Print the resulting hill analysis
            bases = [min(imu_data[hill[0]], imu_data[hill[2]]) for hill in hills]
            # plt.hlines(*[bases, time_data[lefts], time_data[rights]], color=GRAPH_COLORS[value[i][0]][1])
            # plt.plot(time_data[subset[0] : subset[1]], imu_data[subset[0] : subset[1]], color=GRAPH_COLORS[value[i][0]][0])
            # plt.plot(time_data[global_peaks], imu_data[global_peaks], "x", color="Red")
            # plt.plot(time_data[troughs], imu_data[troughs], "x", color="Green")

            # for step in steps:
            #     plt.axvspan(time_data[step[0][0]], time_data[step[-1][2]], color=GRAPH_COLORS[value[i][0]][0], alpha=0.33)

            direct = "Left" if i == 0 else "Right"
            plt.xlabel("Time (s)")
            plt.ylabel("Acceleration (m/s²)")
            plt.title(f"{direct} Data Hills for Patient {GRAPH}")
            plt.show()

            if subset == [0, len(time_data)-1]:
                subset = -1
        all_steps[key] = this_steps
    return all_steps



def main():
    # Non-absoluted UG dictionary
    global non_ab_ug_dict

    # Dictionary of IMU data. {Patient ID: WYAD}
    ug_dict = {}
    non_ab_ug_dict = {}
    # Step Ground Truth (not Game Theory). {Patient ID: [[Step Height], [Time]]}
    step_gt = {}

    """ Read IMU data into ug_dict and ground truth into step_gt """
    for file in os.listdir(data_dir):
        with open(data_dir + os.sep + file) as patient_file:
            # UG Data: UltiGesture IMU data
            ug_data = list(csv.reader(patient_file))
            # This was the first; it has seen everything (Unix time)
            first = int(ug_data[6][1])

            # Test code to only analyze a specific patient
            if ug_data[0][1] != GRAPH:
                continue

            # WYAD: Walking Y Accelerometer Data [["Left", [IMU Readings], [Time]], ["Right", [IMU Readings], [Time]]]
            wyad, non_ab_wyad = read_file(ug_data, first)      
            non_ab_ug_dict[ug_data[0][1]] = non_ab_wyad
            ug_dict[ug_data[0][1]] = wyad

            # Read associated ground truth of the patient.
            step_gt[ug_data[0][1]] = read_gt(ug_data[0][1], first)

            # Generate graph of unanalyzed data
            # graph(wyad)
            # start_time = int((datetime.strptime(ug_data[5][1][1:], "%Y%m%d%H%M%S%f") - datetime(1970, 1, 1)) / timedelta(milliseconds=1))
    
    steps = hill_analysis(ug_dict, [1000, 1200])
    # steps = hill_analysis(ug_dict)
    
    # Perform integration analysis
    for key, value in steps.items():
        # break

        predicted_steps = []
        gt_steps = []
        for i in range(2):
            this_gt = step_gt[key]
            for step in value[i]:
                if len(step) < 3:
                    continue
                # From the 1st peak to the 3rd peak
                air_time_span = [step[0][1], step[2][1]]

                # From the 1st trough to the 3rd trough
                # air_time_span = [step[0][0], step[1][2]]

                air_time = non_ab_ug_dict[key][i][2][air_time_span[0] : air_time_span[1]]

                xy_accel = ug_dict[key][i][3][step[0][0]]
                start_accel = [xy_accel[0], non_ab_ug_dict[key][i][2][step[0][0]], xy_accel[1]]
                magnitude = np.linalg.norm(start_accel)
                norm_start_accel = [[a / magnitude for a in start_accel]]
                desired_accel = [[0, 1, 0]]
                
                # scipy rotation matrix solution
                # Calculate rotation matrix from initial offset of acceleration readings
                rotation_matrix, rssd = R.align_vectors(desired_accel, norm_start_accel)

                # Then, do matrix.apply(vector) in order to rotate the accelerometer readings
                raw_air_time_acc = non_ab_ug_dict[key][i][1][air_time_span[0] : air_time_span[1]].flatten()
                first = raw_air_time_acc[0]
                air_time_acc = [rotation_matrix.apply([[ug_dict[key][i][3][j][0], raw_air_time_acc[j - air_time_span[0]], ug_dict[key][i][3][j][1]]])[0][1] + 9.8 for j in range(air_time_span[0], air_time_span[1])]

                # numpy rotation matrix solution
                # rotation_matrix = rotation_matrix_from_vectors(norm_start_accel, desired_accel)

                # raw_air_time_acc = non_ab_ug_dict[key][i][1][air_time_span[0] : air_time_span[1]].flatten()
                # first = raw_air_time_acc[0]
                # air_time_acc = [rotation_matrix.dot([ug_dict[key][i][3][j][0], raw_air_time_acc[j - air_time_span[0]], ug_dict[key][i][3][j][1]])[1] + 9.8 for j in range(air_time_span[0], air_time_span[1])]

                # Use rotation matrices and double integration to get accurate air time
                velocity = integral(air_time_acc, air_time, initial=0)
                height = integral(velocity, air_time, initial=0)
                # plt.plot(air_time, raw_air_time_acc, label="Raw")
                # plt.plot(air_time, height, label="Height")
                # plt.plot(air_time, velocity, label="Velocity")
                # plt.plot(air_time, air_time_acc, label="Acceleration")
                # plt.xlabel("Time (s)")
                # plt.ylabel("Acceleration (m/s²)")
                # plt.legend()
                # plt.show()

                cur = 0
                # print(f"Calculated Height of Step: {max(height) * 39.37}")
                while this_gt[cur][1] < ug_dict[key][i][2][step[0][0]] or this_gt[cur][1] > ug_dict[key][i][2][air_time_span[1]]:
                    prev_cur = cur
                    # print(this_gt[cur][1])
                    # print(ug_dict[key][i][2][step[0][0]])
                    # print(ug_dict[key][i][2][air_time_span[1]])
                    # print(cur)
                    cur += 1
                    if cur >= len(this_gt):
                        cur = prev_cur
                        break
                if (max(height) * 39.37) < 25:
                    predicted_steps.append(max(height) * 39.37)
                    gt_steps.append(this_gt[cur][0])
                    print(f"Actual Height of Step: {this_gt[cur][0]}")
                    print(f"Predicted Height of Step: {max(height) * 39.37}")
        print(f"Patient {key}:")
        print(f"MAE of Double Integration: {mean_absolute_error(predicted_steps, gt_steps)}")

        plt.scatter(gt_steps, predicted_steps)
        plt.ylabel("DIA Predicted Height")
        plt.xlabel("Ground Truth Height")
        plt.title(f"Patient {key} Double Integration of Acceleration Results")
        plt.show()


                

    
    
    
    
if __name__ == "__main__":
    main()