import requests
import json
import matplotlib.pyplot as plt
import numpy as np
import csv
import time
import pandas as pd
from PIL import Image
from io import BytesIO
import cv2

step_THRESHOLD = 4.5
stair_THRESHOLD = 18
crab_THRESHOLD = 100
turn_THRESHOLD = 300

def get_length_by_id(df, id_value): # Function to get step Length by ID
    """
    Retrieve the length of a step based on a given ID from a DataFrame.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing the step length data.
        id_value (int): The ID value to search for in the DataFrame.
    
    Returns:
        float or None: The length value if the ID is found, otherwise None.
    
    This function looks up the length of a step based on the given ID.
    If the ID exists in the DataFrame, the corresponding length is returned.
    """
    length_value = df.loc[df['ID'] == id_value, 'Length'].values
    if len(length_value) > 0:
        return length_value[0]
    else:
        return None

def init_files(t): # Init csv file for 3D graph of Position
    with open("data.csv", 'w') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=["x_value", "y_value", "z_value", "step_type"])
        csv_writer.writeheader()
    with open("sensors.csv", 'w') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=["time_acc", "corr_z", "x_acc", "corr_y", "x_gyro", "combined"])
        csv_writer.writeheader()
        info = {
            "time_acc": 1000*t,
            "corr_z": 0,
            "x_acc": 0,
            "corr_y": 0,
            "x_gyro": 0,
            "combined": 0
        }
        csv_writer.writerow(info)

def update_data_file(step_type, position): # Update data csv file for 3D graph of Position
    with open("data.csv", 'a') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=["x_value", "y_value", "z_value", "step_type"])
        x_value = round(position[2],2)
        y_value = round(position[0],2)
        z_value = round(position[1],2)
        info = {
            "x_value": x_value,
            "y_value": y_value,
            "z_value": z_value,
            "step_type": step_type
        }
        csv_writer.writerow(info)

def update_sensors_file(time_acc, corr_z, x_acc, corr_y, x_gyro, combined): # Update sensors csv file for 2D graphs of sensors' data
    with open("sensors.csv", 'a') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=["time_acc", "corr_z", "x_acc", "corr_y", "x_gyro", "combined"])
        for i in range(len(time_acc)):
            info = {
                "time_acc": time_acc[i],
                "corr_z": corr_z[i],
                "x_acc": x_acc[i],
                "corr_y": corr_y[i],
                "x_gyro": x_gyro[i],
                "combined": combined[i]
            }
            csv_writer.writerow(info)

def step_to(step_type, step_size, facing, position, stair_size_vertical, stair_size_horizontal): # Calculates updated position according to last movement kind
    if (step_type == 0.1): # Forward step
        if(facing == 0): # x+
            position[2] += step_size
        elif(facing == 1): # y-
            position[0] -= step_size
        elif(facing == 2): # x-
            position[2] -= step_size
        else: # y+
            position[0] += step_size
    elif (step_type == 0.2): # Backward step
        if(facing == 0): # x+
            position[2] -= step_size
        elif(facing == 1): # y-
            position[0] += step_size
        elif(facing == 2): # x-
            position[2] += step_size
        else: # y+
            position[0] -= step_size
    elif (step_type == 1.1): # Stair down
        position[1] -= stair_size_vertical
        if(facing == 0): # x+
            position[2] += stair_size_horizontal
        elif(facing == 1): # y-
            position[0] -= stair_size_horizontal
        elif(facing == 2): # x-
            position[2] -= stair_size_horizontal
        else: # y+
            position[0] += stair_size_horizontal
    elif (step_type == 1.2): # Stair up
        position[1] += stair_size_vertical
        if(facing == 0): # x+
            position[2] += stair_size_horizontal
        elif(facing == 1): # y-
            position[0] -= stair_size_horizontal
        elif(facing == 2): # x-
            position[2] -= stair_size_horizontal
        else: # y+
            position[0] += stair_size_horizontal
    elif (step_type == 2.1): # Crab Right
        if(facing == 0): # x+
            position[0] -= step_size
        elif(facing == 1): # y-
            position[2] -= step_size
        elif(facing == 2): # x-
            position[0] += step_size
        else: # y+
            position[2] += step_size
    elif (step_type == 2.2): # Crab Left
        if(facing == 0): # x+
            position[0] += step_size
        elif(facing == 1): # y-
            position[2] += step_size
        elif(facing == 2): # x-
            position[0] -= step_size
        else: # y+
            position[2] -= step_size

def getArrays(data_dict): # Extract sensors' data
    # Get sensors' data
    acc_arr = np.array(data_dict['lin_accel']['data'], dtype=object)
    gyro_arr = np.array(data_dict['gyro']['data'], dtype=object)

    acc_arr = np.concatenate((acc_arr[:,:1], np.array(acc_arr[:,1].tolist())), axis=1)
    gyro_arr = np.concatenate((gyro_arr[:,:1], np.array(gyro_arr[:,1].tolist())), axis=1)
    return acc_arr, gyro_arr

def getRelevantInfo(sensor_arr, sensor_data, last_time_sensor, t):
    time_column_sensor = sensor_arr[:, 0]
    # Create a boolean mask for the conditions (time from now, eliminate duplicated values)
    mask_sensor = (time_column_sensor > last_time_sensor) & (time_column_sensor >= 1000 * t)
    # Apply the mask to filter the rows
    sensor_arr = sensor_arr[mask_sensor]
    sensor_data = np.concatenate((sensor_data, sensor_arr), axis=0)
    if(len(sensor_data) > 0):
        last_time_sensor = sensor_data[-1][0]
    return sensor_arr, sensor_data, last_time_sensor

def plot_graph(x, y, title, two_graphs=0, x_two=0, y_two=0, title2=0, combined = 0): #Gets 1 or 2 (optional) graphs parameters and plots
    fig, ax = plt.subplots()
    if(combined):
        values = [step_THRESHOLD, stair_THRESHOLD, crab_THRESHOLD, turn_THRESHOLD]
        colors = ['orange', 'red', 'green', 'yellow']
        names = ['step', 'stair', 'crab', 'turn']
        for value, color, name in zip(values, colors, names):
            ax.axhline(y=value, color=color, label=f'{name}')
    plt.title(title, fontsize=20)
    plt.grid(True)
    plt.xlabel("Time [ms]", fontsize=20)
    plt.ylabel("acceleration [m/s^2]", fontsize=20)
    ax.plot(x, y, label=title) #, marker="."
    if(two_graphs==1):
        plt.plot(x_two, y_two, label=title2)
    fig.legend()

def image_snap(image_count, snap_times, session, ip_user):

    # Fetch the image data
    response1 = session.get(f'http://{ip_user}:8080/shot.jpg')
    snap_times = np.append(snap_times, 1000*time.time())

    # Check if the request was successful
    if response1.status_code == 200:
        img_data1 = response1.content
        # Open the image data using Pillow
        image1 = Image.open(BytesIO(img_data1))
        
        # Save the image to a file
        # image1.save(f"zz{image_count}.jpg") # UNCOMMENT IF YOU WANT THE IMAGES ON YOUR PC

        print(f"Images {image_count} saved successfully at time {time.time()}")
        
        return np.array(image1), snap_times#, np.array(image2)

    else:
        print("Failed to retrieve the image.")

def compute_optical_flow_step(image1, image2): # step detection
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Calculate dense optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    return flow

def get_mean_flow(flow, box_definitions): # step detection
    mean_flows = []
    for (start_x_ratio, start_y_ratio, width_ratio, height_ratio, direction) in box_definitions:
        height, width, _ = flow.shape
        start_x = int(start_x_ratio * width)
        start_y = int(start_y_ratio * height)
        box_width = int(width_ratio * width)
        box_height = int(height_ratio * height)
        
        # Extract flow in the specified box
        flow_box = flow[start_y:start_y + box_height, start_x:start_x + box_width]
        
        # Calculate the average flow in the specified direction
        if direction == 'vertical_up':
            avg_flow = -np.mean(flow_box[..., 1])  # vertical flow (y direction), inverted for upward
        elif direction == 'vertical_down':
            avg_flow = np.mean(flow_box[..., 1])  # vertical flow (y direction)
        elif direction == 'horizontal_left':
            avg_flow = -np.mean(flow_box[..., 0])  # horizontal flow (x direction), inverted for leftward
        elif direction == 'horizontal_right':
            avg_flow = np.mean(flow_box[..., 0])  # horizontal flow (x direction)
        
        mean_flows.append(avg_flow)
        print(f"the avg flow in box {direction} is : {avg_flow}")
    return np.sum(mean_flows)

def visualize_flow_with_arrows(image1, flow, box_definitions, step=16): # step detection
    height, width, _ = image1.shape
    y, x = np.mgrid[step//2:height:step, step//2:width:step]
    
    fx, fy = flow[y, x, 0], flow[y, x, 1]

    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    plt.quiver(x, y, fx, fy, color='red', angles='xy', scale_units='xy', scale=1)
    
    # Draw the boxes
    for (start_x_ratio, start_y_ratio, width_ratio, height_ratio, _) in box_definitions:
        start_x = int(start_x_ratio * width)
        start_y = int(start_y_ratio * height)
        box_width = int(width_ratio * width)
        box_height = int(height_ratio * height)
        plt.gca().add_patch(plt.Rectangle((start_x, start_y), box_width, box_height, 
                                          edgecolor='blue', facecolor='none', lw=2))
    
    plt.title('Optical Flow with Arrows')
    plt.axis('off')
    plt.show()

def step_optical_flow(image1, image2): # step detection
    # Compute optical flow
    flow = compute_optical_flow_step(image1, image2)
    aligned_image2 = align_images(image1, image2)
    aligned_mean_flow_sum = 0

    # Box definitions: (start_x_ratio, start_y_ratio, width_ratio, height_ratio, direction)
    box_definitions = [
        (0.35, 0.05, 0.30, 0.30, 'vertical_up'),
        (0.35, 0.65, 0.30, 0.30, 'vertical_down'),
        (0.10, 0.20, 0.20, 0.60, 'horizontal_left'),
        (0.70, 0.20, 0.20, 0.60, 'horizontal_right')
    ]

    mean_flow_sum = get_mean_flow(flow, box_definitions)

    if (aligned_image2 is not None):
        aligned_flow = compute_optical_flow_step(image1, aligned_image2)
        aligned_mean_flow_sum = get_mean_flow(aligned_flow, box_definitions)

    mean_flow_sum = mean_flow_sum + aligned_mean_flow_sum

    # Classify movement based on the sum of mean flows
    if mean_flow_sum > 0:
        movement = 'one step forward'
        print(f'Movement detected: {movement}')
        return 0.1
    else:
        movement = 'one step backward'
        print(f'Movement detected: {movement}')
        return 0.2

    # Visualize optical flow
    #visualize_flow_with_arrows(image1, flow, box_definitions)

def compute_optical_flow_stair(image1, image2): # stair detection
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Calculate dense optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Get the dimensions of the flow
    h, w = flow.shape[:2]

    # Create a grid of points at which to plot the vectors
    step = 16
    y, x = np.mgrid[step/2:h:step, step/2:w:step].astype(int)

    return flow

def analyze_vertical_movement(flow): # stair detection
    # Extract vertical component of flow
    vertical_flow = flow[:, :, 1]

    # Calculate mean of vertical flow
    mean_vertical_flow = np.mean(vertical_flow)
    print("mvf : ", mean_vertical_flow)
    return mean_vertical_flow

def stair_optical_flow(image1, image2): # stair detection
    # Compute optical flow
    flow = compute_optical_flow_stair(image1, image2)
    aligned_image2 = align_images(image1, image2)
    aligned_mean_vertical_flow = 0

    mean_vertical_flow = analyze_vertical_movement(flow)

    if(aligned_image2 is not None):
        aligned_flow = compute_optical_flow_stair(image1, aligned_image2)
        aligned_mean_vertical_flow = analyze_vertical_movement(aligned_flow)

    mean_vertical_flow = mean_vertical_flow + aligned_mean_vertical_flow

    # Classify movement
    if mean_vertical_flow < 0:
        movement = 'one stair down'
        print(f'Movement detected: {movement}')
        return 1.1
    else:
        movement = 'one stair up'
        print(f'Movement detected: {movement}')
        return 1.2

def align_images(image1, image2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect SIFT keypoints and descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # Check if descriptors are valid
    if descriptors1 is None or descriptors2 is None:
        print("One of the images has no descriptors.")
        return None

    if descriptors1.dtype != np.float32 or descriptors2.dtype != np.float32:
        print("Descriptors are not of type float32.")
        return None

    # Match descriptors using FLANN matcher
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test to get good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good_matches.append(m)

    draw_params = dict(matchColor=(255, 0, 0), singlePointColor=(0, 0, 255), flags=cv2.DrawMatchesFlags_DEFAULT)
    img3 = cv2.drawMatchesKnn(gray1, keypoints1, gray2, keypoints2, matches, None, **draw_params)
    # plt.imshow(img3,)
    # plt.show()

    # Extract location of good matches
    points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
    points2 = np.zeros((len(good_matches), 2), dtype=np.float32)

    for i, match in enumerate(good_matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    if len(good_matches) < 4:
        return None

    # Find homography matrix
    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

    # Ensure h is a 3x3 matrix
    if h is None or h.shape != (3, 3):
        # Debugging: Check homography matrix
        if h is not None:
            print(f"Homography matrix shape: {h.shape}")
            print(f"Homography matrix type: {h.dtype}")
        else:
            print("Homography matrix is None")
        return None

    # Use homography to warp image2 to align with image1
    height, width, channels = image1.shape
    aligned_image2 = cv2.warpPerspective(image2, h, (width, height))

    return aligned_image2

def main():
    # IP numbers of different connections
    ip_shahar = "172.20.10.3"
    #ip_iddo = "192.168.71.114"
    ip_iddo = "192.168.45.114"
    #ip_iddo = "192.168.71.141"

    # DEFINE CURRENT IP
    ip_user = ip_iddo

    # URLs by chosen IP
    #url = f'http://{ip_shahar}:8080/sensors.json' # Shahar's Iphone
    url = f'http://{ip_user}:8080/sensors.json' # Iddo's phone
    #url='http://192.168.1.98:8080/sensors.json' # Iddo's Home
    #url='http://192.168.1.37:8080/sensors.json' # Shahar's Home
    url_images=f'http://{ip_user}:8080/jsfs.html'

    # Init arrays for time samples of the sensors
    time_acc = np.empty(0)
    time_gyro = np.empty(0)

    # Init arrays for acc values
    x_acc = np.empty(0)
    y_acc = np.empty(0)
    z_acc = np.empty(0)

    # Init arrays for gyro values
    x_gyro = np.empty(0)

    # Init arrays for correlated values
    corr_x = np.empty(0)
    corr_x_up = np.empty(0)
    corr_y = np.empty(0)
    corr_z = np.empty(0)
    corr_z_original = np.empty(0)
    corr_z_down = np.empty(0)
    x_gyro_original = np.empty(0)
    x_gyro_data_original = np.empty(0)
    y_gyro = np.empty(0)
    combined = np.empty(0)
    combined_total = np.empty(0)
    combined_up = np.empty(0)
    combined_total_up = np.empty(0)
    corrbined_x = np.empty(0)
    current_step_z = np.empty(0)
    current_step_z_time = np.empty(0)
    z_data_uncorrelated = np.empty(0)
    snap_times = np.empty(0)

    # Init stairs constants
    stair_size_vertical = 0.18
    stair_size_horizontal = 0.35

    # Init time trackers
    last_time_acc = -1
    last_time_gyro = -1

    # Init step counters
    steps = 0
    right_turns = 0
    left_turns = 0
    stairs = 0
    right_crab = 0
    left_crab = 0
    image_count = 1

    # Init Thresholds
    step_forwards_THRESHOLD = 9
    step_backwards_THRESHOLD = 7
    stair_upwards_THRESHOLD = 40
    stair_downwards_THRESHOLD = 32
    global step_THRESHOLD
    global stair_THRESHOLD
    global crab_THRESHOLD
    global turn_THRESHOLD

    # Init flags
    zcd = False
    active_step = False
    active_turn = False
    active_stair = False
    active_crab = False
    image_taken = False
    under_bottom = True
    above_top = False
    under_down = True
    above_up = False

    # Init location statuses
    facing = 0 # Facing: 0=straight (x+), 1=right (y-), 2=back (x-), 3=left (y+) 
    position = [0,0,0] # Start position

    # Init number of main loop iterations
    main_loops = 10

    # Init session
    session = requests.Session()

    # Store one-step samples for correlation
    one_step_time_z = [1708075953693, 1708075953757, 1708075953847, 1708075953890, 1708075953956, 1708075954022, 1708075954093, 1708075954135, 1708075954201, 1708075954263, 1708075954327, 1708075954399, 1708075954464, 1708075954524, 1708075954587]
    one_step_z = [0.0019283295, -0.31461418, -0.38757485, -0.5615046, -0.82077456, -0.70846677, -0.22448683, 1.3322704, 1.9106109, 2.1643274, 1.8346863, 0.41018885, 0.6492244, 0.4345014, -0.011036992]
    one_step_time_x = [1709056702415, 1709056702473, 1709056702530, 1709056702586, 1709056702643, 1709056702700, 1709056702811, 1709056702867, 1709056702927, 1709056702983, 1709056703038, 1709056703093, 1709056703149]
    one_step_x_down = [0.49236965, -1.1043682, -2.3008747, -3.8264022, -2.6175961, -1.9925332, 0.6265106, 2.408804, 4.023218, 3.2465353, 1.5863628, 0.44496822, -1.266716]
    one_step_time_x_up = [1709056698939, 1709056698939, 1709056698998, 1709056699054, 1709056699109, 1709056699166, 1709056699222, 1709056699278, 1709056699335, 1709056699447, 1709056699504, 1709056699562]
    one_step_x_up = [-1.5733957, -0.7198553, 1.225626, 4.1041822, 4.209552, 1.5392923, -0.64115334, -3.075348, -3.065333, -2.2356324, -0.7757168, -0.16170979]
    one_step_time_y1 = [1716895278191, 1716895278902, 1716895279334, 1716895280185]
    one_step_y1 = [0.06601444, -0.99877065, 2.0134683, 0.0020478368]
    one_step_time_y = [1716895278191, 1716895278369, 1716895278547, 1716895278725, 1716895278903, 1716895279011, 1716895279119, 1716895279227, 1716895279335, 1716895279477, 1716895279619, 1716895279761, 1716895279903, 1716895280045, 1716895280187]
    one_step_y = [0.06601444, -0.20018183, -0.4663781,  -0.73257438, -0.99877065, -0.24571091,  0.50734883,  1.26040856,  2.0134683,  1.67823156, 1.34299481, 1.00775807, 0.67252132, 0.33728458, 0.00204784]


    print("Please enter your ID:")
    id = int(input())
    # Create a DataFrame from the CSV file
    df = pd.read_csv('StepLengthEstimation.csv')
    step_size = get_length_by_id(df,id)/100
    t = time.time()
    print("first time : ", t)
    init_files(t)
    update_data_file("Init", position)
    image1, snap_times = image_snap(0, snap_times, session, ip_user)
    for i in range(main_loops): # Number of iterations of the program
        print("the current iteration is : " , i)
        acc_data = np.empty((0,4))
        gyro_data = np.empty((0,4)) # Init Data arrays for Acc and Gyro (block)
        while (len(acc_data) < 16): # Build a single block of info, of length >= 16 of Accelerometer 

            # Connect and get info from phone's sensors
            response = requests.get(url)
            data = response.content
            data_dict = json.loads(data)

            # Extract sensors' data
            acc_arr, gyro_arr = getArrays(data_dict)
            
            # Filter for only relevant samples out of the last 'get' command
            acc_arr, acc_data, last_time_acc = getRelevantInfo(acc_arr, acc_data, last_time_acc, t)
            gyro_arr, gyro_data, last_time_gyro = getRelevantInfo(gyro_arr, gyro_data, last_time_gyro, t)

        if(len(acc_data) > len(gyro_data)):
            acc_data = acc_data[:len(gyro_data),:]
        else:
            gyro_data = gyro_data[:len(acc_data),:]

        correlatd_x = np.correlate(acc_data[: ,1], one_step_x_down, "same")
        correlatd_x_up = np.correlate(acc_data[: ,1], one_step_x_up, "same")

        (correlatd_x)[(correlatd_x >= -35) & (correlatd_x <= 35)] = 0
        (correlatd_x_up)[(correlatd_x_up >= -35) & (correlatd_x_up <= 35)] = 0
        corrbined_x = np.append(corrbined_x, correlatd_x_up - correlatd_x)

        # Concatenate sensor_data (last block) to sensor (the accumulating array)
        (acc_data[:,1])[(acc_data[:,1] >= -3) & (acc_data[:,1] <= 3)] = 0 # Zero the noise for acc_data_x (block)
        x_acc = np.concatenate((x_acc, acc_data[: ,1]), axis=0)
        y_acc = np.concatenate((y_acc, acc_data[: ,2]), axis=0)
        z_acc = np.concatenate((z_acc, acc_data[: ,3]), axis=0)

        x_gyro_original = np.concatenate((x_gyro_original, gyro_data[:,1]), axis=0)
        x_gyro_data_original = gyro_data[:,1]
        
        (gyro_data[:,1])[(gyro_data[:,1] >= -1.2) & (gyro_data[:,1] <= 1.2)] = 0 # Zero the noise for gyro_data_x (block)
        x_gyro = np.concatenate((x_gyro, gyro_data[:,1]), axis=0)
        y_gyro = np.concatenate((y_gyro, gyro_data[:,2]), axis=0)
        
        # Concatenate sensors' time from last block to the acculumating time arrays
        time_acc = np.concatenate((time_acc, acc_data[: ,0]), axis=0)
        time_gyro = np.concatenate((time_gyro, gyro_data[: ,0]), axis=0)

        # Correlate desired arrays with some function
        correlated_z = np.correlate(acc_data[: ,3], one_step_z, "same")
        correlated_z_down = np.correlate(acc_data[: ,3], one_step_x_down, "same")
        correlated_y = np.correlate(acc_data[: ,2], one_step_y, "same")

        z_data_uncorrelated = acc_data[:,3] # Save an original version of z_acc block

        acc_data[:,3] = correlated_z
        correlated_z[(correlated_z >= -(step_THRESHOLD-0.4)) & (correlated_z <= (step_THRESHOLD-0.4))] = 0 # Zero the noise for correlated z (block)
        correlated_y[(correlated_y >= -12.5) & (correlated_y <= 12.5)] = 0 # Zero the noise for correlated y (block)

        correlatd_x = correlatd_x[:len(gyro_data)]
        correlatd_x_up = correlatd_x_up[:len(gyro_data)]

        # Concatenate new blocks to accumulating arrays
        corr_x = np.concatenate((corr_x, correlatd_x), axis=0)
        corr_x_up = np.concatenate((corr_x_up, correlatd_x_up), axis=0)
        corr_z = np.concatenate((corr_z, correlated_z), axis=0)
        corr_z_original = np.concatenate((corr_z_original, acc_data[:,3]), axis=0)
        corr_z_down = np.concatenate((corr_z_down, correlated_z_down), axis=0)
        corr_y = np.concatenate((corr_y, correlated_y), axis=0)

        #acc_data = acc_data[:len(gyro_data)] #WARNINGGGGGGGGGGGGGG
        combined = 200*np.abs(gyro_data[:,1]) + 0.01*x_gyro_data_original + 0.01*acc_data[:,3] + correlated_z + 10*np.abs(correlated_y) + 6*acc_data[: ,1]
        combined_up = 60*np.abs(gyro_data[:,1]) + correlatd_x_up
        combined_total = np.concatenate((combined_total, combined), axis=0)
        combined_total_up = np.concatenate((combined_total_up, combined_up), axis=0)

        update_sensors_file(acc_data[:,0], correlated_z, acc_data[:,1], correlated_y, x_gyro_data_original, combined)

        prev_value = 0
        for i in range(len(correlatd_x)):
            current_value = combined[i]

            current_step_z = np.append(current_step_z, z_data_uncorrelated[i])
            current_step_z_time = np.append(current_step_z_time, acc_data[: ,0][i])

            # Check if a certain step is detected
            if(current_value >= 0): # Positive value
                if(active_step and zcd): # Any step detected
                    #print("Any step detected : ", current_value, " at time : ", gyro_data[:,0][i])
                    if(active_turn): # Turn detected
                        if(turn_value > 0): # Turn Left detected
                            step_type = "left_turn"
                            left_turns += 1
                            facing = (facing - 1) % 4
                            print("Left turn detected : ", left_turns)
                        else: # Turn right detected
                            step_type = "right_turn"
                            right_turns += 1
                            facing = (facing + 1) % 4
                            print("Right turn detected : ", right_turns)
                    elif (active_crab): # Crab detected
                        if(crab_value > 0): # Crab Right detected
                            step_type = "right_crab"
                            right_crab += 1
                            step_to(2.1, step_size, facing, position, stair_size_vertical, stair_size_horizontal)
                            print("Right Crab detected : ", right_crab)
                        else: # Crab Left detected
                            step_type = "left_crab"
                            left_crab += 1
                            step_to(2.2, step_size, facing, position, stair_size_vertical, stair_size_horizontal)
                            print("Left Crab detected : ", left_crab)
                    elif (active_stair): # Stair detected
                        stairs += 1
                        if(under_down):
                            step_type = "stair_down"
                            print("UNDER DOWN --------------------")
                            step_to(1.1, step_size, facing, position, stair_size_vertical, stair_size_horizontal) #backwards
                        elif(above_up):
                            step_type = "stair_up"
                            print("ABOVE UP --------------------")
                            step_to(1.2, step_size, facing, position, stair_size_vertical, stair_size_horizontal) #backwards
                        else:
                            print("PICTURE OF STAIRS----------")
                            stair_dir = stair_optical_flow(image0, image1)
                            if(stair_dir == 1.2):
                                step_type = "stair_up"
                            else:
                                step_type = "stair_down"
                            step_to(stair_dir, step_size, facing, position, stair_size_vertical, stair_size_horizontal)
                        print("Stair detected : ", stairs)
                    else: # Step detected
                        steps += 1
                        if(under_bottom):
                            step_type = "step_backward"
                            print("UNDER BOTTOM !!!!!!!!!!!!!!!!!!!!!!!11")
                            step_to(0.2, step_size, facing, position, stair_size_vertical, stair_size_horizontal) #backwards
                        elif(above_top):
                            step_type = "step_forward"
                            print("ABOVE TOP !!!!!!!!!!!!!!!!!!!!!!!11")
                            step_to(0.1, step_size, facing, position, stair_size_vertical, stair_size_horizontal) #forwards
                        else:
                            print("PICTURE OF STEP!!!!!!!!!!!!!!!!!!!!11")
                            step_dir = step_optical_flow(image0, image1)
                            step_to(step_dir, step_size, facing, position, stair_size_vertical, stair_size_horizontal)
                            if(step_dir == 0.1):
                                step_type = "step_forward"
                            else:
                                step_type = "step_backward"
                        print("Step detected : ", steps)
                    active_turn = False
                    active_stair = False
                    active_step = False
                    active_crab = False
                    image_taken = False
                    under_bottom = True
                    above_top = False
                    under_down = True
                    above_up = False
                    update_data_file(step_type, position)
                if(current_value >= step_THRESHOLD): # Larger than step_THRESHOLD
                    active_step = True
                    if(current_value >= step_forwards_THRESHOLD):
                        above_top = True
                    if(current_value >= step_backwards_THRESHOLD):
                        under_bottom = False
                if(current_value >= turn_THRESHOLD): # Larger than Turn Threshold
                    active_turn = True
                    #turn_index = i
                    turn_value = gyro_data[:,1][i]
                elif(current_value >= crab_THRESHOLD): # Larger than Crab Threshold
                    active_crab = True
                    crab_value = correlated_y[i]
                elif(current_value >= stair_THRESHOLD): # Larger than Stair Threshold
                    if(current_value >= stair_upwards_THRESHOLD):
                        above_up = True
                    if(current_value >= stair_downwards_THRESHOLD):
                        under_down = False
                    active_stair = True
                zcd = False
            else: # Negative value
                if(not image_taken and active_step):
                    print(f"image snap at {time.time()}")
                    image0 = image1
                    image1, snap_times = image_snap(image_count, snap_times, session, ip_user)
                    image_count = image_count + 1
                    image_taken = True
                zcd = True
                #if (current_value < -100):
                #    active_stair = True
            prev_value = current_value


    #plot_graph(time_acc, y_acc, "y_acc")
    #plot_graph(time_acc, corrbined_x, "corrbined x")
    #plot_graph(time_acc, x_acc, "accelerometer[x]", 1, time_acc, corrbined_x, "corrbined x")
    #plot_graph(time_acc, x_acc, "accelerometer[x]", 1, time_acc, corr_x, "corr x")
    #plot_graph(time_acc, x_acc, "accelerometer[x]", 1, time_acc, corr_x_up, "corr x up")
    #plot_graph(time_acc, corr_x, "corr x", 1, time_acc, corr_x_up, "corr x up")
    #plot_graph(time_acc, z_acc, "z_acc")
    #plot_graph(time_acc, corr_y, "corr_y", 1, time_acc, x_acc, "x_acc")
    #plot_graph(time_acc, corr_y, "corr_y", 1, time_acc, corr_z, "corr_z")
    #plot_graph(time_acc, y_acc, "y_acc", 1, time_acc, z_acc, "z_acc")
    #plot_graph(time_acc, corr_x, "corr x")
    #plot_graph(time_acc, corr_x_up, "corr x up")
    #plot_graph(time_acc, corr_y, "correlated accelerometer[y]")
    #plot_graph(one_step_time_y, one_step_y, "one_step_y")
    #plot_graph(time_acc, corr_y, "corr_y", 1, time_acc, combined_total, "combined")
    #plot_graph(time_acc, z_acc, "z_acc")
    #plot_graph(time_acc, corr_z, "correlated accelerometer[z]")
    #plot_graph(time_acc, corr_z_original, "correlated accelerometer[z] original")
    #plot_graph(time_gyro, corr_z_down, "corr z down")
    #plot_graph(time_gyro, x_gyro, "gyroscope[x]")
    #plot_graph(time_gyro, x_gyro_original, "x gyro original")
    #plot_graph(time_gyro, y_gyro, "y_gyro")
    #plot_graph(time_gyro, 6*x_gyro + x_acc, "Combined")
    #plot_graph(time_gyro, 60*x_gyro + corr_x, "Combined 2")
    #plot_graph(time_gyro, x_gyro, "x gyro")
    #plot_graph(time_gyro, x_gyro_original, "x gyro combined")
    plot_graph(time_gyro, combined_total, "Combined Weighted Graph", combined=1) #Real Combined
    #plot_graph(time_gyro, corr_z, "Correlated z acc")
    #plot_graph(time_gyro, z_acc, "z acc")
    #plot_graph(time_gyro, z_acc-0.13, "z acc - 0.13")
    #plot_graph(time_gyro, corr_x, "Correlated x acc")
    #plot_graph(time_gyro, combined_total_up, "Combined up")
    #plot_graph(one_step_time_x, one_step_x_down, "one step x down")
    #plot_graph(one_step_time_x_up, one_step_x_up, "one step x up")
    #plot_graph(one_step_time_z, one_step_z, "one step z")
    #plot_graph(time_acc, x_gyro, "Gyro x / time", 1, time_acc, corr_z, "Corr z / time")
    #plot_graph(time_acc, x_gyro, "Gyro x / time", 1, time_acc, np.correlate(corr_z, x_gyro, "same"), "cor^2 z / time")

    plt.show()

if __name__ == "__main__":
    main()
