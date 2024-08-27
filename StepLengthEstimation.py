import requests
import json
import matplotlib.pyplot as plt
import numpy as np
import csv
import time
import scipy
import pykalman

def init_file(): # Init csv file for saving individuals' step length
    with open("Final Project/StepLengthEstimation.csv", 'w') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=["ID", "Length"])
        csv_writer.writeheader()

def update_file(id, length): # Update csv file for saving individuals' step length
    with open("Final Project/StepLengthEstimation.csv", 'a') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=["ID", "Length"])
        ID = id
        Length = length
        info = {
            "ID": ID,
            "Length": Length
        }

        csv_writer.writerow(info)
        #print(x_value, y_value, z_value)

def plot_graph(x, y, title, two_graphs=0, x_two=0, y_two=0, title2=0): #Gets 1 or 2 (optional) graphs parameters and plots
    plt.title(title)
    #plt.xlabel('time')
    #plt.ylabel('correlated z')
    plt.grid(True)
    plt.xlabel("Time [s]")
    plt.ylabel("acceleration [m/s^2]")
    plt.plot(x, y, label=title) #, marker="."
    if(two_graphs==1):
        plt.plot(x_two, y_two, label=title2)
    plt.legend()
    plt.figure()
    plt.savefig(f"{title}.png")

def getArrays(data_dict): # Extract sensors' data
    # Get sensors' data
    acc_arr = np.array(data_dict['lin_accel']['data'], dtype=object)

    # Make it a Numpy array of [time, sensor_data_1 , sensor_data_2 {optional}, sensor_data_3 {optional}]
    acc_arr = np.concatenate((acc_arr[:,:1], np.array(acc_arr[:,1].tolist())), axis=1)

    return acc_arr

def getRelevantInfo(t, sensor_arr, sensor_data, last_time_sensor):
    time_column_sensor = sensor_arr[:, 0]
    # Create a boolean mask for the conditions (time from now, eliminate duplicated values)
    mask_sensor = (time_column_sensor > last_time_sensor) & (time_column_sensor >= 1000 * t)
    # Apply the mask to filter the rows
    sensor_arr = sensor_arr[mask_sensor]
    sensor_data = np.concatenate((sensor_data, sensor_arr), axis=0)
    if(len(sensor_data) > 0):
        last_time_sensor = sensor_data[-1][0]
    return sensor_data, last_time_sensor

def blockBuilding(t, url, acc_data, last_time_acc):
    while (len(acc_data) < 15): # Build a single block of info, of length >= 15 of Accelerometer
        # Next rows: connect and get info from phone's sensors
        response = requests.get(url)
        data = response.content
        data_dict = json.loads(data)
        # Extract sensors' data
        acc_arr = getArrays(data_dict)
        # Filter for only relevant samples out of the last 'get' command
        acc_data, last_time_acc = getRelevantInfo(t, acc_arr, acc_data, last_time_acc)
    return acc_data, last_time_acc
    

def loop(t, url, time_acc, z_acc, corr_z, last_time_acc, steps, zcd, active_step, one_step_z, route):
    for i in range(2*route): # Number of iterations of the program
        #print("the current iteration is : " , i)
        acc_data = np.empty((0,4))
        acc_data, last_time_acc = blockBuilding(t, url, acc_data, last_time_acc)
        # Concatenate sensor_data (last block) to sensor (the accumulating array)
        z_acc = np.concatenate((z_acc, acc_data[: ,3]), axis=0)
        
        # Concatenate sensors' time from last block to the acculumating time arrays
        time_acc = np.concatenate((time_acc, acc_data[: ,0]), axis=0)

        # Correlate desired arrays with some function
        correlated_z = np.correlate(acc_data[: ,3], one_step_z, "same")

        z_data_uncorrelated = acc_data[:,3] # Save an original version of z_acc block

        acc_data[:,3] = correlated_z

        # Concatenate new blocks to accumulating arrays
        corr_z = np.concatenate((corr_z, correlated_z), axis=0)

        for i in range(len(correlated_z)):
            current_value = correlated_z[i]
            # Check if a certain step is detected
            if(current_value >= 0): # Positive value
                if(active_step and zcd): # step detected
                    steps += 1
                    print("Step detected : ", steps)
                    active_step = False
                zcd = False
                if(current_value >= 5): # Larger than 5
                    active_step = True
            else: # Negative value
                zcd = True
    return steps, time_acc, corr_z

def main():
    print("Please enter your ID:")
    id = input()
    print("Please enter the maximal route length [meters] you can walk forward now:")
    route = int(input())
    print("Pleae walk in a straight route of ", route, " meters in 3 seconds")
    print("3...")
    time.sleep(1)
    print("2...")
    time.sleep(1)
    print("1...")
    time.sleep(1)
    print("WALK!")
    t = time.time()
    time_acc = np.empty(0)
    z_acc = np.empty(0)
    corr_z = np.empty(0)
    last_time_acc = -1
    steps = 0
    zcd = False
    active_step = False
    url='http://172.20.10.3:8080/sensors.json' # Shahar's Iphone
    url='http://192.168.71.114:8080/sensors.json' # Iddo's phone
    #url='http://192.168.1.98:8080/sensors.json' # Iddo's Home
    #url='http://192.168.1.37:8080/sensors.json' # Shahar's Home
    one_step_z = [0.0019283295, -0.31461418, -0.38757485, -0.5615046, -0.82077456, -0.70846677, -0.22448683, 
                  1.3322704, 1.9106109, 2.1643274, 1.8346863, 0.41018885, 0.6492244, 0.4345014, -0.011036992]
    init_file()
    steps, time_acc, corr_z = loop(t, url, time_acc, z_acc, corr_z, last_time_acc, steps, zcd, active_step, one_step_z, route)
    if(steps == 0):
        print("You didn't walk :( try to walk again")
    else:
        step_length = route*100/steps
        print("the step length of ", id, " is : ", step_length, " cm")
        update_file(id, step_length)
        plot_graph(time_acc, corr_z, "corr z")
        plt.show()
    tacc = ""
    zacc = ""
    for i in range(len(time_acc)):
        tacc += f"{time_acc[i]},"
        zacc += f"{corr_z[i]},"
    print(f'tacc : {tacc}')
    print(f'zacc : {zacc}')


if __name__=="__main__":
    main()