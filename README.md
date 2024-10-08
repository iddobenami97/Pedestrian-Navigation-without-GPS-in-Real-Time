# Pedestrian-Navigation-without-GPS-in-Real-Time
Final Project of our Electrical Engineering degree.

# Pedestrian Trajectory Detection and Visualization

This project detects and classifies different types of pedestrian steps (step forward/backward, stair up/down, crab step left/right, turn left/right) based on accelerometer, gyroscope, and camera data from a chest-mounted mobile phone. The output is a CSV file with the classified steps, which is then used to visualize the trajectory in 3D.

## Prerequisites

1. **Android Device with IP Webcam**: Install the IP Webcam app from the Google Play Store.
2. **Python Environment**: Ensure the following Python libraries are installed:
    - `requests`
    - `json`
    - `matplotlib`
    - `numpy`
    - `scipy`
    - `pykalman`
    - `pandas`
    - `matplotlib.animation`
    - `mpl_toolkits.mplot3d`

## Setup Instructions

1. **Mobile Device Connection**:
    - Ensure your Android device and PC are connected to the same network.
    - Open the IP Webcam app and start the server.
    - Note the IP address displayed on the app screen (e.g., `http://192.168.0.101:8080/`). You will use this in the code.

2. **Step Length Estimation**:
    - Run the `StepLengthEstimation.py` script to initialize and estimate your step length:
      ```bash
      python StepLengthEstimation.py
      ```
    - Enter your ID and follow the instructions to walk a specific distance. This data is saved in the `StepLengthEstimation.csv` file.

3. **Running the Main Program**:
    - Ensure the `projipwebcam.py` script has the correct IP address set in the `url` variable. Update it according to the IP from the IP Webcam app.
    - Start the 3D animation by running:
      ```bash
      python animate_updated.py
      ```
    - With the animation running, execute the main program:
      ```bash
      python projipwebcam.py
      ```
    - Enter the same ID you used for the step length estimation.

4. **Output**:
    - The output will be a 3D visualization of your trajectory with step classifications and directions annotated.

## Code Overview

- **StepLengthEstimation.py**: Initializes the step length estimation file and computes individual step lengths using accelerometer data.
- **animate_updated.py**: Generates a 3D real-time visualization of the pedestrian trajectory.
- **projipwebcam.py**: The main program for detecting steps and directions using accelerometer, gyroscope, and camera data.

## Important Notes

- The IP address in both `StepLengthEstimation.py` and `projipwebcam.py` must be updated according to your mobile device’s IP Webcam address.
- Run `StepLengthEstimation.py` first to estimate step length, which will be used in subsequent runs of `projipwebcam.py`.
- The program `projipwebcam.py` will save a csv file for the sensors documentation on the user's PC.

## Example Workflow

1. Connect your mobile device and start the IP Webcam server.
2. Estimate step length:
    ```bash
    python StepLengthEstimation.py
    ```
3. Visualize the trajectory:
    ```bash
    python animate_updated.py
    ```
4. Run the main detection algorithm:
    ```bash
    python projipwebcam.py
    ```

## Troubleshooting

- If the 3D graph does not update correctly, ensure that both scripts are running simultaneously and the IP addresses are correct.
- Ensure the mobile device is securely mounted to the pedestrian's chest in landscape mode for accurate data capture.
