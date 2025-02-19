# Comparative Study of Traditional and Deep Learning Feature Detectors and Matchers for Land Vehicle Monocular Visual Odometry

This repository contains the code and data for our study, which compares traditional and deep learning-based feature detection and matching methods in the context of land vehicle monocular visual odometry (VO). The work evaluates performance in terms of computational efficiency, trajectory estimation accuracy, and robustness across various driving scenarios.

The code will be uploaded upon acceptance

## Abstract

Visual odometry (VO) enables autonomous navigation for land vehicles by estimating motion through visual data. This study compares classical detectors like SIFT and ORB with deep learning-based approaches such as SuperPoint and DISK. Similarly, traditional matchers like FLANN and SNN are evaluated alongside deep learning matchers such as ADALAM and LIGHTGLUE. The study emphasizes trade-offs between computational efficiency and accuracy under different motion dynamics and environmental conditions.

## Tested Data
### Scenario 1: Straight-Line Motion
- **Description**: Vehicle moves in a straight line at a consistent speed of ~30 km/h, covering a distance of 91 meters.
- **Visualization**:  
  ![Scenario 1](assets/gifs/sequence_1.gif)

### Scenario 2: Turning Motion
- **Description**: Vehicle slows down to ~10 km/h for a turn, then accelerates to ~15 km/h, covering a distance of 17.4 meters.
- **Visualization**:  
  ![Scenario 2](assets/gifs/sequence_2.gif)

### Scenario 3: Deceleration to Stop
- **Description**: Vehicle decelerates from ~32.5 km/h to 0, covering a distance of 37 meters.
- **Visualization**:  
  ![Scenario 3](assets/gifs/sequence_3.gif)

## How to Use
1. Clone the repository:
   ```bash
    git clone https://github.com/olaayman/MonocularVisualOdometry.git

2. Make conda environment:
    ```bash
      conda create --name mono_vo python=3.11.7
      conda activate mono_vo
      pip install -r requirements.txt

3. Run Code:
  - To test one feature detector and feature matcher run the test_single_combination.py file
    ```bash
      python3 test_single_combination.py
    ```
  
  - To run tests for many combinations together run the test_multiple_combinations.py file
    ```bash
      python3 test_multiple_combinations.py
    ```

## List of Parameters

In both file test_single_combination.py and test_multiple_combinations.py you will find parameters in the main file that you will need to adapt to your data and tests.

- K: camera matrix
- body_to_cam_angles: euler angles to transform from body frame to camera frame
- poses_timestamps: path to reference timestamps in csv format
- data_dir: directory were the images are (note the images are assumed to be named with the timestamp it was taken at)
- poses_dir: path to the ground truth poses in txt format for the flatten pose matrix
- visualize_matches: bool to choose to visualize the feature matches or not
- downsample: downsampling factor for the data
- images_freq: images frame per seconds
- gps_freq: the reference frequency
- start: the index of the image to start at
- feature_detectors: one or a list of feature detectors
- feature_matchers: one or a list of feature matchers


## **Data Format**  

The data consists of three main components: images, ground truth data, and pose matrices. 

A sample 10 seconds data sequence is provided in the data folder and for the images you can download the zip file from [HERE](https://drive.google.com/file/d/1P9ADb4e0ufNIFuVOTsE-JF_kcP4VIHTu/view?usp=sharing) and extract the images in the `data/images/` folder as shown below 

### **1. Folder Structure**
```
data/
│── images/                     # Contains PNG images named with the ROS timestamp (e.g., `1708453200.123.png`)
│── novatel-reference.csv        # CSV file with ground truth data
│── poses.txt                    # Text file with pose matrices (one per row)
```
### **2. Ground Truth Data (novatel-reference.csv)**  


| Column Name | Data Type | Description |
|-------------|----------|-------------|
| `timestamp` | `float`  | ROS timestamp of the recorded data |
| `header.stamp.secs` |  `int` | ROS timestamp.seconds of the recorded data |
| `header.stamp.nsecs` |  `int` | ROS timestamp.nanoseconds of the recorded data |
| `latitude`  | `float`  | Latitude (degrees) |
| `longitude` | `float`  | Longitude (degrees) |
| `altitude`  | `float`  | Altitude (meters) |
| `roll`      | `float`  | Vehicle roll angle (degrees) |
| `pitch`     | `float`  | Vehicle pitch angle (degrees) |
| `azimuth`       | `float`  | Vehicle azimuth angle (degrees) |

### **4. Pose Data (poses.txt)**  OPTIONAL
- **condition**: If you don't have the ground truth data in the previous format and you have it in a pose matrix format then you can use this format but you need to provide timestamps in a csv file.

- **Location**: `data/poses.txt`  
- **Format**: Plain text  
- **Description**: Each line contains a **flattened** 3×4 transformation matrix (row-major order), representing the camera pose at a given timestamp.  
- **Data Format**:  
  ```
  r11 r12 r13 t1 r21 r22 r23 t2 r31 r32 r33 t3
  ```
  - `r_ij`: Rotation matrix elements  
  - `t_i`: Translation vector components (in meters)  


- **Example Entry:**
  ```
  0.030541 0.017734 0.999376 379405.423 -0.999534 0.000640 0.030535 4899765.551 -0.000099 -0.999843 0.017745 72.649475
  ```

## **NavINST Dataset**  

The main dataset used in this project is NavINST dataset. which is a multi-sensory dataset from various road trajectories in Kingston, ON, and Calgary, AB, Canada. 

For more deatails about the sensors and the trajectories you can visit [NavINST Dataset](https://navinst.github.io/) 

To download the dataset visit [Download NavINST Dataset](https://www.frdr-dfdr.ca/repo/dataset/8f8b4d74-3264-4e7d-b851-8a7b4e804cf8)

To visualize, intereact and helpful scripts visit [NavINST Dataset Code](https://github.com/NavInst/dataset)


## **Acknowledgements**
- **ComputerVision** – Used as a baseline for visual odometry implementation.  
  - Source: https://github.com/niconielsen32/ComputerVision/tree/master/VisualOdometry  

## **Citation**
#### If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{Elmaghraby2025,
  author    = {Ola Elmaghraby and Paulo Ricardo Marques Araujo and Shaza I. Kaoud Abdelaziz and Aboelmagd Noureldin},
  title     = {Comparative Study of Traditional and Deep Learning Feature Detectors and Matchers for Land Vehicle Monocular Visual Odometry},
  booktitle = {Proceedings of the IEEE International Systems Conference (SysCon)},
  year      = {2025}
}
```
