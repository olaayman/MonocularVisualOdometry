#%%
import os
import numpy as np
import cv2

from visualization import plotting
from visualization.video import play_trip

from tqdm import tqdm
import csv
import time
##Kornia imports
import kornia as K
import kornia.feature as KF
import kornia.geometry.epipolar as KG
import torch
import matplotlib.pyplot as plt
import pandas as pd
from transformers import AutoImageProcessor, SuperPointForKeypointDetection, SuperPointConfig
from PIL import Image
from scipy.spatial.transform import Rotation as Rot
import pyproj

device = K.utils.get_cuda_or_mps_device_if_available()
print(device)

class VisualOdometry():
    def __init__(self, data_dir, poses_dir,poses_csv_filepath, start, end, downsample, images_freq, gps_freq, feature_detector, feature_matcher, calibration_file=None, K=None, P=None, speed_dir=None, odometer_freq=None, units="m/s", visualization=False, body_to_cam_angles = [-90.000, -90.0, 0.000]):
        """
        Initializes the Visual Odometry object.

        Parameters
        ----------
        data_dir (str): The directory containing the images
        poses_dir (str): The directory containing the poses
        start (int): The start frame
        end (int): The end frame
        downsample (int): The downsample factor
        images_freq (int): The images frequency
        gps_freq (int): The GPS frequency
        feature_detector (str): The feature detector to use
        feature_matcher (str): The feature matcher to use
        calibration_file (str): The calibration file path
        K (ndarray): The intrinsic parameters
        P (ndarray): The projection matrix
        speed_dir (str): The directory containing the speeds
        odometer_freq (int): The odometer frequency
        units (str): The speed units (m/s or km/h) default is m/s
        visualization (bool): Whether to visualize the matches or not (default is False)
        """
        
        self.start = start
        self.end = end
        self.downsample = downsample
        self.feature_detector = feature_detector
        self.feature_matcher = feature_matcher
        self.gps_freq = gps_freq
        self.images_freq = images_freq
        self.odometer_freq = odometer_freq
        self.body_to_cam_angles = body_to_cam_angles

        if calibration_file is not None:
            self.K, self.P = self._load_calib(calibration_file)
        else:
            self.K = K
            self.P = P

        self.gt_poses, self.pose_secs, self.pose_nsecs = self._load_poses(poses_dir,poses_csv_filepath)
        self.image_paths, self.image_secs, self.image_nsecs = self._load_image_paths(data_dir, start, end)
        if speed_dir is not None:
            self.speeds, self.speed_secs, self.speed_nsecs = self._load_speeds(speed_dir, units)
        else:
            self.speeds = None
            self.speed_secs = None
            self.speed_nsecs = None
        self.sync_and_downsample()
        self.orb = cv2.ORB_create(3000)
        self.orb = cv2.ORB_create(2048)
        self.visualization = visualization
        

    @staticmethod
    def _load_calib(filepath):
        """
        Loads the calibration of the camera
        Parameters
        ----------
        filepath (str): The file path to the camera file

        Returns
        -------
        K (ndarray): Intrinsic parameters
        P (ndarray): Projection matrix
        """
        # filepath = "C:/Users/olael/Downloads/KITTI_dataset/sequences/01/calib.txt"
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('P2:'):
                    print(line[3:].strip())  # strip() is used to remove leading/trailing whitespace
                    params = np.fromstring(line[3:].strip(), dtype=np.float64, sep=' ')
                    P = np.reshape(params, (3, 4))
                    print("p",P)
                    K = P[0:3, 0:3]

     
        return K, P

    def gps_to_cartesian(self, lat, lon, height):
        # Define the WGS84 ellipsoid
        wgs84 = pyproj.CRS("EPSG:4326")  # WGS84
        # Define the UTM projection for Zone 18
        utm = pyproj.CRS("EPSG:32618")  # UTM Zone 18N
        
        # Create a transformer object
        transformer = pyproj.Transformer.from_crs(wgs84, utm)
        
        # Transform latitude and longitude to UTM coordinates
        easting, northing = transformer.transform(lat, lon)
        
        return np.array([easting, northing, height])
    
    def ref_to_camera_frame(self, R):
        R_new = Rot.from_euler('ZXY', self.body_to_cam_angles, degrees=True).as_matrix()        
        R_trans = R_new @ R
        return R_trans

    def reference_to_poses(self, ref_data):
        """
        Converts the reference data in lat, lon, alt, roll, pitch and azimuth to poses matrix

        Parameters
        ----------
        ref_data (ndarray): The reference data in lat, lon, alt, roll, pitch and azimuth

        Returns
        -------
        poses (ndarray): The poses matrix
        """
        poses = []
        latitudes = ref_data['latitude']
        longitudes = ref_data['longitude']
        heights = ref_data['height']
        rolls = ref_data['roll']
        pitches = ref_data['pitch']
        azis = ref_data['azimuth']
        # Iterate over the data
        for i in range(len(latitudes)):
            # Convert GPS to Cartesian coordinates
            position = self.gps_to_cartesian(latitudes[i], longitudes[i], heights[i])
          
            R = Rot.from_euler('ZXY', [azis[i], pitches[i], rolls[i] ], degrees=True).as_matrix()
            R_cam = self.ref_to_camera_frame(R)
            Rt = np.hstack((R_cam, np.array([position[0],position[1], position[2]]).reshape(3, 1)))
            Rt = np.vstack((Rt, [0, 0, 0, 1]))
            poses.append(Rt)
        return poses
        
    def _load_poses(self, filepath, poses_csv_filepath):
        """
        Loads the GT poses

        Parameters
        ----------
        filepath (str): The file path to the poses file
        poses_csv_filepath (str): The file path to the poses timestamps file

        Returns
        -------
        poses (ndarray): The GT poses
        poses_secs (list): The GT poses seconds timestamps
        poses_nsecs (list): The GT poses nano seconds timestamps
        """
        ## load the timestamps and the poses
        novatel_reference = pd.read_csv(poses_csv_filepath)
        pose_secs = novatel_reference['header.stamp.secs']
        pose_nsecs = novatel_reference['header.stamp.nsecs']

        if filepath.endswith('.txt'):
            poses = []
            with open(filepath, 'r') as f:
                for line in f.readlines():
                    T = np.fromstring(line, dtype=np.float64, sep=' ')
                    T = T.reshape(3, 4)
                    T = np.vstack((T, [0, 0, 0, 1]))
                    poses.append(T)
        elif filepath.endswith('.csv'):
            poses = self.reference_to_poses(novatel_reference)
            

        return poses, pose_secs, pose_nsecs
    

    @staticmethod
    def _load_image_paths(filepath, start, end):
        """
        Loads the images

        Parameters
        ----------
        filepath (str): The file path to image dir

        Returns
        -------
        images_paths (list): The image paths
        images_secs (list): The image seconds timestamps
        images_nsecs (list): The image nano seconds timestamps
        """
        print("in load image paths")
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        # image name is the timestamp of the image
        image_secs = [int(os.path.basename(file).split('.')[0]) for file in image_paths]
        image_nsecs = [int(os.path.basename(file).split('.')[1]) for file in image_paths]
        # cut the images to the start and end
        image_paths = image_paths[start:end+1]
        image_secs = image_secs[start:end+1]
        image_nsecs = image_nsecs[start:end+1]
        return image_paths, image_secs, image_nsecs


    @staticmethod
    def _load_speeds(filepath, units):
        """
        Loads the speeds
        
        Parameters
        ----------
        filepath (str): The file path to the speeds file in csv format with the speeds and timestamps
        units (str): The speed units (km/h or m/s)
        
        Returns
        -------
        speeds (ndarray): The speeds in m/s
        speed_secs (list): The speeds seconds timestamps
        speed_nsecs (list): The speeds nano seconds timestamps
        """
        # Load the speeds from csv file the speed is in km/h and the column name is data

        odometer_data = pd.read_csv(filepath)
        speed_secs = odometer_data['header.stamp.secs'].values
        speed_nsecs = odometer_data['header.stamp.nsecs'].values
        if units == "km/h":
            speeds = odometer_data['data'].values * 1000 / 3600 # convert to m/s
        elif units == "m/s":
            speeds = odometer_data['data'].values
        return speeds, speed_secs, speed_nsecs
      

    def sync_and_downsample(self):
        """
        Synchronizes the images and the poses and downsamples the data.

        This method synchronizes the timestamps of the images and poses, and then downsamples the data
        based on the specified frequency. It performs the following steps:
        1. Finds the closest pose timestamp for each image timestamp.
        2. Syncs the data by selecting the corresponding poses and images.
        3. Downsamples the data based on the specified frequency.
        
        Returns:
            None
        """

        '''SYNC THE DATA with image frequency 30 Hz'''
        pose_indices = []
        pose_time = np.array(self.pose_secs) + np.array(self.pose_nsecs) *1e-9
        images_time = np.array(self.image_secs) + np.array(self.image_nsecs) * 1e-9
      
        sort_image_indices = np.argsort(images_time)
        images_time = images_time[sort_image_indices]
        self.image_secs = np.array(self.image_secs)[sort_image_indices]
        self.image_nsecs = np.array(self.image_nsecs)[sort_image_indices]
        self.image_paths = np.array(self.image_paths)[sort_image_indices]

        start_time = max(pose_time[0], images_time[0])
        end_time = min(pose_time[-1], images_time[-1])
        print("start_time", start_time)
        print("end_time", end_time)
        # get first index of the time list that is more than start time
        
        pose_idx_time_check = np.where((pose_time > start_time) & (pose_time < end_time))[0]
        image_idx_time_check = np.where((images_time > start_time) & (images_time < end_time))[0]

        self.pose_secs = np.array(self.pose_secs)[pose_idx_time_check]
        self.pose_nsecs = np.array(self.pose_nsecs)[pose_idx_time_check]
        self.image_secs = np.array(self.image_secs)[image_idx_time_check]
        self.image_nsecs = np.array(self.image_nsecs)[image_idx_time_check]
        self.image_paths = np.array(self.image_paths)[image_idx_time_check]
        self.gt_poses = np.array(self.gt_poses)[pose_idx_time_check]
        pose_time = pose_time[pose_idx_time_check]
        images_time = images_time[image_idx_time_check]

        for imgs_t in images_time:
            pose_idx = np.argmin(np.abs(pose_time - imgs_t))
            pose_indices.append(pose_idx)
        
        # Sync the data
        self.gt_poses = [self.gt_poses[i] for i in pose_indices]

        # Downsample the data
        # self.gt_poses = self.gt_poses[self.start:self.end]
        # self.image_paths = self.image_paths[self.start:self.end]
        self.gt_poses = self.gt_poses[::self.downsample]
        self.image_paths = self.image_paths[::self.downsample]

        # make translation of the first pose origin and subtract it from all the poses only last coloumn
        self.gt_poses = np.array(self.gt_poses)
        temp = self.gt_poses[0][:,3].copy()
        temp[3] = 0.0
        for i in range(len(self.gt_poses)):
            self.gt_poses[i][:,3] = self.gt_poses[i][:,3].copy() - temp


        '''SYNC THE DATA with odometer frequency 16.666 Hz'''
        # pose_indices = []
        # image_indices = []
        # speed_time = np.array(self.speed_secs) + np.array(self.speed_nsecs) *1e-9
        # pose_time = np.array(self.pose_secs) + np.array(self.pose_nsecs) *1e-9
        # images_time = np.array(self.image_secs) + np.array(self.image_nsecs) * 1e-9
        # start_time = max(speed_time[0], pose_time[0], images_time[0])
        # # get first index of the time list that is more than start time
        
        # speed_idx_more_start_time = np.where(speed_time > start_time)[0]
        # pose_idx_more_start_time = np.where(pose_time > start_time)[0]
        # image_idx_more_start_time = np.where(images_time > start_time)[0]
        # print("speed_idx_more_start_time", speed_idx_more_start_time)
        # print("pose_idx_more_start_time", pose_idx_more_start_time)
        # print("image_idx_more_start_time", image_idx_more_start_time)
        # self.speed_secs = np.array(self.speed_secs)[speed_idx_more_start_time]
        # self.speed_nsecs = np.array(self.speed_nsecs)[speed_idx_more_start_time]
        # self.pose_secs = np.array(self.pose_secs)[pose_idx_more_start_time]
        # self.pose_nsecs = np.array(self.pose_nsecs)[pose_idx_more_start_time]
        # self.image_secs = np.array(self.image_secs)[image_idx_more_start_time]
        # self.image_nsecs = np.array(self.image_nsecs)[image_idx_more_start_time]
        # self.speeds = np.array(self.speeds)[speed_idx_more_start_time]
        # self.image_paths = np.array(self.image_paths)[image_idx_more_start_time]
        # self.gt_poses = np.array(self.gt_poses)[pose_idx_more_start_time]
        # speed_time = speed_time[speed_idx_more_start_time]
        # pose_time = pose_time[pose_idx_more_start_time]
        # images_time = images_time[image_idx_more_start_time]

        # # make sure all image nsecs are 9 digits and if more 
        # for i, (speed_sec, speed_nsec) in enumerate(zip(self.speed_secs, self.speed_nsecs)):
        #     print("speed_sec", speed_sec, "speed_nsec", speed_nsec)
        #     pose_idx = np.argmin(np.abs(pose_time - speed_time[i]))
        #     image_idx = np.argmin(np.abs(images_time - speed_time[i]))
        #     # pose_idx = np.argmin(np.abs(np.array(self.pose_secs) - speed_sec) + np.abs(np.array(self.pose_nsecs) - speed_nsec))
        #     # image_idx = np.argmin(np.abs(np.array(self.image_secs) - speed_sec) + np.abs(np.array(self.image_nsecs) - speed_nsec))
        #     print("image_secs", self.image_secs[image_idx], "images_nsces", self.image_nsecs[image_idx])
        #     print("pose_idx", pose_idx, "image_idx", image_idx)
        #     pose_indices.append(pose_idx)
        #     image_indices.append(image_idx)
        
        # # Sync the data
        # self.gt_poses = [self.gt_poses[i] for i in pose_indices]
        # self.image_paths = [self.image_paths[i] for i in image_indices]

        # # Downsample the data
        # ratio = self.images_freq / self.odometer_freq
        # start = int(self.start // ratio)
        # end = int(self.end // ratio)+1
        # print("start", start, "end", end)
        # self.gt_poses = self.gt_poses[start:end]
        # self.image_paths = self.image_paths[start:end]
        # self.speeds = self.speeds[start:end]
        # self.gt_poses = self.gt_poses[::self.downsample]
        # self.image_paths = self.image_paths[::self.downsample]
        # self.speeds = self.speeds[::self.downsample]
        # print("len speeds",len(self.speeds))
        # print("speeds", self.speeds[:10])
        return
    

    @staticmethod
    def _hamming_distance_matrix(desc1, desc2):
        """
        Compute the Hamming distance matrix between two sets of binary descriptors.

        Args:
            desc1 (np.array): First set of binary descriptors with shape (N, D).
            desc2 (np.array): Second set of binary descriptors with shape (M, D).

        Returns:
            torch.Tensor: Hamming distance matrix with shape (N, M).
        """
        # Ensure the descriptors are binary
        assert desc1.dtype == np.uint8 and desc2.dtype == np.uint8, "Descriptors must be of type np.uint8"

        # Compute pairwise Hamming distances using vectorized operations
        desc1_expanded = np.expand_dims(desc1, axis=1)  # Shape: (N, 1, D)
        desc2_expanded = np.expand_dims(desc2, axis=0)  # Shape: (1, M, D)
        
        # Compute pairwise XOR and count the number of differing bits
        xor_result = np.bitwise_xor(desc1_expanded, desc2_expanded)  # Shape: (N, M, D)
        distance_matrix = np.sum(np.unpackbits(xor_result, axis=2), axis=2).astype(np.int16)  # Shape: (N, M)

        # Convert back to torch tensor
        distance_matrix = torch.tensor(distance_matrix, device=device)

        return distance_matrix
        

    @staticmethod
    def _form_transf(R, t):
        """
        Makes a transformation matrix from the given rotation matrix and translation vector

        Parameters
        ----------
        R (ndarray): The rotation matrix
        t (list): The translation vector

        Returns
        -------
        T (ndarray): The transformation matrix
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def read_image(self, i):
        """
        Reads the i'th image

        Parameters
        ----------
        i (int): The image index

        Returns
        -------
        current_img: The current image
        previous_img: The previous image
        """
        current_img = None
        previous_img = None
        if self.feature_detector == "disk":
            torch.cuda.empty_cache()
            current_img = K.io.load_image(self.image_paths[i], K.io.ImageLoadType.RGB32, device=device)[None, ...] 
            previous_img = K.io.load_image(self.image_paths[i - 1], K.io.ImageLoadType.RGB32, device=device)[None, ...]
            #images = images.detach().cpu().numpy()
        elif self.feature_detector == "orb" or self.feature_detector == "sift":
            current_img = cv2.imread(self.image_paths[i], cv2.IMREAD_GRAYSCALE) 
            previous_img = cv2.imread(self.image_paths[i - 1], cv2.IMREAD_GRAYSCALE)
            D = np.array([-0.227871168414825, 0.067457701518732, 0, 0, -0.0100296183963231])
            current_img = cv2.undistort(current_img, self.K , D)
            previous_img = cv2.undistort(previous_img, self.K , D)
        elif self.feature_detector == "superpoint":
            current_img = cv2.imread(self.image_paths[i])
            current_img = cv2.cvtColor(current_img, cv2.COLOR_BGR2RGB)
            previous_img = cv2.imread(self.image_paths[i - 1])
            previous_img = cv2.cvtColor(previous_img, cv2.COLOR_BGR2RGB)
            D = np.array([-0.227871168414825, 0.067457701518732, 0, 0, -0.0100296183963231])
            current_img = cv2.undistort(current_img, self.K , D)
            previous_img = cv2.undistort(previous_img, self.K , D)
   

        else:
            print("feature detector not supported")
            return []
        return current_img, previous_img
    
    def feature_detection(self, current_img, previous_img):
        """
        Detects the features in the current and previous images

        Parameters
        ----------
        current_img: The current image
        previous_img: The previous image

        Returns
        -------
        kp1 (ndarray): The keypoints in the previous image
        des1 (ndarray): The descriptors in the previous image
        kp2 (ndarray): The keypoints in the current image
        des2 (ndarray): The descriptors in the current image
        kps1 (tensor): The keypoints in the previous image
        descs1 (tensor): The descriptors in the previous image
        kps2 (tensor): The keypoints in the current image
        descs2 (tensor): The descriptors in the current image
        dm (tensor): The hamming distance matrix
        """

        kp1, des1, kp2, des2, kps1, descs1, kps2, descs2, dm = None, None, None, None, None , None, None, None, None


        if self.feature_detector == "orb": ## orb is a binary descriptor so it does not work with lightglue
            # Find the keypoints and descriptors with ORB
            kp1, des1 = self.orb.detectAndCompute(previous_img, None)
            kp2, des2 = self.orb.detectAndCompute(current_img, None)
            descs1 = torch.tensor(des1, device=device)
            descs2 = torch.tensor(des2, device=device)
            kps1 = torch.tensor(np.float32([kp1[m].pt for m in range(len(kp1))]), device=device)
            kps2 = torch.tensor(np.float32([kp2[m].pt for m in range(len(kp2))]), device=device)


            #get the hamming distance
            if self.feature_matcher != "flann":
                dm = self._hamming_distance_matrix(des1, des2)
           
        elif self.feature_detector == "disk":
            num_features = 2048
            disk = KF.DISK.from_pretrained("depth").to(device)



            with torch.inference_mode():
                inp = torch.cat([torch.tensor(previous_img, device=device), torch.tensor(current_img, device=device)], dim=0)
                features1, features2 = disk(inp, num_features, pad_if_not_divisible=True)
                kps1, descs1 = features1.keypoints, features1.descriptors
                kps2, descs2 = features2.keypoints, features2.descriptors
                des1 = descs1.detach().cpu().numpy()
                des2 = descs2.detach().cpu().numpy()
                kp1 = kps1.detach().cpu().numpy()
                kp2 = kps2.detach().cpu().numpy()
      
        elif self.feature_detector == "sift":
            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(previous_img, None)
            kp2, des2 = sift.detectAndCompute(current_img, None)
            des1 = np.float32(des1)
            des2 = np.float32(des2)
            descs1 = torch.tensor(np.float32(des1), device=device)
            descs2 = torch.tensor(np.float32(des2), device=device)
            kps1 = torch.tensor(np.float32([kp1[m].pt for m in range(len(kp1))]), device=device)
            kps2 = torch.tensor(np.float32([kp2[m].pt for m in range(len(kp2))]), device=device)
            
        elif self.feature_detector == "superpoint":

            images = [torch.tensor(previous_img, device=device), torch.tensor(current_img, device=device)]
            #images = torch.cat([previous_img, current_img], dim=0).to(device)
            configuration = SuperPointConfig(max_keypoints=2048)
            # Initializing a model from the superpoint style configuration
            processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint")
            model = SuperPointForKeypointDetection(configuration).from_pretrained("magic-leap-community/superpoint").to(device)

            #inputs = processor(images, return_tensors="pt").to(device)
            inputs = processor(images,size={"height": 720, "width": 1280}, return_tensors="pt").to(device)
            outputs = model(**inputs)

            idx1 = outputs.mask[0].nonzero().squeeze()
            kps1 = outputs.keypoints[0][idx1]
            descs1 = outputs.descriptors[0][idx1]
            idx2 = outputs.mask[1].nonzero().squeeze()
            kps2 = outputs.keypoints[1][idx2]
            descs2 = outputs.descriptors[1][idx2]

            # to visualize the keypoints on the image uncomment the following lines
            # for keypoint in zip(kps1):
            #     keypoint_x, keypoint_y = int(keypoint[0].item()), int(keypoint[1].item())
            #     image_1 = np.array(previous_img)
            #     image = cv2.circle(image_1, (keypoint_x, keypoint_y), 2)
            #     plt.imshow(image)
            #     plt.show()

            des1 = descs1.detach().cpu().numpy()
            des2 = descs2.detach().cpu().numpy()
            kp1 = kps1.detach().cpu().numpy()
            kp2 = kps2.detach().cpu().numpy()

        return kp1, des1, kp2, des2, kps1, descs1, kps2, descs2, dm

    def get_matches(self, i):
        """
        This function detect and compute keypoints and descriptors from the i-1'th and i'th image using the class orb object

        Parameters
        ----------
        i (int): The current frame

        Returns
        -------
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image
        """
        '''READ IMAGES ONLINE'''
        current_img, previous_img = self.read_image(i)
        
        '''FEATURE DETECTION'''
        hw1 = torch.tensor(previous_img.shape[2:], device=device)
        hw2 = torch.tensor(current_img.shape[2:], device=device)
        kp1, des1, kp2, des2, kps1, descs1, kps2, descs2, dm = self.feature_detection(current_img, previous_img)

        '''MATCHING'''
        idxs = None
        if self.feature_matcher == "bf":
            # Initialize the Brute-Force Matcher with Hamming distance
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

            # Perform knn matching (k=2 for Lowe's ratio test)
            des1 = np.uint8(des1)
            des2 = np.uint8(des2)
            print("des1 shape", des1.shape, des1)
            matches = bf.knnMatch(des1, des2, k=2)

            # Apply the ratio test to filter matches
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:  # 0.75 is a commonly used ratio threshold
                    good.append(m)

            # Get the image points form the good matches
            if self.feature_detector == "disk":
                q1 = np.float32([kp1[m.queryIdx] for m in good])
                q2 = np.float32([kp2[m.trainIdx] for m in good])
            else:
                q1 = np.float32([kp1[m.queryIdx].pt for m in good])
                q2 = np.float32([kp2[m.trainIdx].pt for m in good])

            return q1, q2
        
        elif self.feature_matcher == "flann":
            # Find matches
            # des1 = np.uint8(des1)
            # des2 = np.uint8(des2)
            if self.feature_detector == "orb":
                FLANN_INDEX_LSH = 6
                index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
            else:
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)

            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
            matches = flann.knnMatch(des1, des2, k=2)

            # # Find the matches there do not have a to high distance
            good = []
            try:
                for m, n in matches:
                    if m.distance <= 0.8 * n.distance:
                        good.append(m)
            except ValueError:
                pass

            
            ''' VISUALIZATION OF THE MATCHES'''
            if self.visualization:
                draw_params = dict(matchColor = -1, # draw matches in green color
                        singlePointColor = None,
                        matchesMask = None, # draw only inliers
                        flags = 2)
                # # current_img = current_img.detach().cpu().numpy()
                # # previous_img = previous_img.detach().cpu().numpy()
                current_img = cv2.imread(self.image_paths[i])
                previous_img = cv2.imread(self.image_paths[i - 1])
                print("current_img", current_img.shape)
                if self.feature_detector == "disk" or self.feature_detector == "superpoint":
                    kp1_cv2 = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in kp1]
                    kp2_cv2 = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in kp2]
                else:
                    kp1_cv2 = kp1
                    kp2_cv2 = kp2

                # get the index of the keypoint with max x
                # index = np.argmax(kp1[:, 0], axis=0)
                # print("good", kp1[index][0], kp1[index][1])

                img3 = cv2.drawMatches(previous_img, kp1_cv2, current_img, kp2_cv2, good[::20] ,None,**draw_params)
                # # img3 = cv2.drawMatches(current_img, kp1, previous_img, kp2, good ,None,**draw_params)
                #img3 = cv2.circle(current_img, (int(kp1[:][0]), int(kp1[:][1])), 2, color=tuple([2 * 255] * 3))

                cv2.imshow("image", img3)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            # Get the image points form the good matches
            if self.feature_detector == "disk" or self.feature_detector == "superpoint":
                q1 = np.float32([kp1[m.queryIdx] for m in good])
                q2 = np.float32([kp2[m.trainIdx] for m in good])
            else:
                q1 = np.float32([kp1[m.queryIdx].pt for m in good])
                q2 = np.float32([kp2[m.trainIdx].pt for m in good])

            return q1, q2

        elif self.feature_matcher == "snn":
            # snn
            dists, idxs = KF.match_snn(descs1, descs2, 0.88, dm=dm)
           
        elif self.feature_matcher == "adalam":
            adalam_config = KF.adalam.get_adalam_default_config()
            adalam_config["force_seed_mnn"] = False
            adalam_config["search_expansion"] = 16
            adalam_config["ransac_iters"] = 256
            lafs1 = KF.laf_from_center_scale_ori(kps1[None], 96 * torch.ones(1, len(kps1), 1, 1, device=device))
            lafs2 = KF.laf_from_center_scale_ori(kps2[None], 96 * torch.ones(1, len(kps2), 1, 1, device=device))

            dists, idxs = KF.match_adalam(descs1, descs2, lafs1, lafs2, hw1=hw1, hw2=hw2, config=adalam_config, dm=dm)

            pass
        elif self.feature_matcher == "fginn":
            lafs1 = KF.laf_from_center_scale_ori(kps1[None], 96 * torch.ones(1, len(kps1), 1, 1, device=device))
            lafs2 = KF.laf_from_center_scale_ori(kps2[None], 96 * torch.ones(1, len(kps2), 1, 1, device=device))
            dists, idxs = KF.match_fginn(descs1, descs2, lafs1, lafs2, dm=dm )

            pass
        elif self.feature_matcher == "smnn":
            dists, idxs = KF.match_smnn(descs1, descs2, 0.98, dm=dm)
            pass
        elif self.feature_matcher == "lightglue":
            if self.feature_detector == "superpoint":
                weights = "superpoint"
            else:
                weights = "disk"
            lg = KF.LightGlue(weights).to(device).eval()
            #lg_matcher = KF.LightGlueMatcher("disk").eval().to(device)
            #onnx_lg = KF.OnnxLightGlue(weights='disk_fp16', device=device)
         
            image0 = {
                "keypoints": kps1[None],
                "descriptors": descs1[None],
                #"image_size": torch.tensor(previous_img.shape[-2:][::-1], device=device).view(1, 2),
                "image_size": torch.tensor([   1280, 720], device=device).view(1, 2),
            }
            image1 = {
                "keypoints": kps2[None],
                "descriptors": descs2[None],
                #"image_size": torch.tensor(current_img.shape[-2:][::-1], device=device).view(1, 2),
                "image_size": torch.tensor([   1280, 720], device=device).view(1, 2),
            }

            # onnx lg
            # out = onnx_lg({"image0": image0, "image1": image1})
            # idxs = out["matches"]

            # lg
            out = lg({"image0": image0, "image1": image1})
            idxs = out["matches"][0]

            # lg_matcher
            # lafs1 = KF.laf_from_center_scale_ori(kps1[None], torch.ones(1, len(kps1), 1, 1, device=device))
            # lafs2 = KF.laf_from_center_scale_ori(kps2[None], torch.ones(1, len(kps2), 1, 1, device=device))
            # dists, idxs = lg_matcher(descs1, descs2, lafs1, lafs2, hw1=hw1, hw2=hw2)

            pass
        else:
            print("feature matcher not supported")
            return [], []
        
        ''' VISUALIZATION OF THE MATCHES'''
        if self.visualization:
            draw_params = dict(matchColor = -1, # draw matches in green color
                        singlePointColor = None,
                        matchesMask = None, # draw only inliers
                        flags = 2)
            # Convert idxs to a list of cv2.DMatch objects
            matches = [cv2.DMatch(_queryIdx=int(idx[0]), _trainIdx=int(idx[1]), _distance=0) for idx in idxs.detach().cpu().numpy()]
            current_img = cv2.imread(self.image_paths[i])
            previous_img = cv2.imread(self.image_paths[i - 1])
            if self.feature_detector == "disk" or self.feature_detector == "superpoint":
                kp1_cv2 = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in kp1]
                kp2_cv2 = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in kp2]
            else:
                kp1_cv2 = kp1
                kp2_cv2 = kp2

            img3 = cv2.drawMatches(previous_img, kp1_cv2, current_img, kp2_cv2, matches[::20] ,None,**draw_params)
            # # img3 = cv2.drawMatches(current_img, kp1, previous_img, kp2, good ,None,**draw_params)
            cv2.imshow("image", img3)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # Get the image points form the good matches
        q1 = kps1[idxs[:, 0]]
        q2 = kps2[idxs[:, 1]]
        return q1.detach().cpu().numpy(), q2.detach().cpu().numpy()
      

    def get_pose(self, q1, q2):
        """
        Calculates the transformation matrix

        Parameters
        ----------
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image

        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix
        """
        # Essential matrix
        E, _ = cv2.findEssentialMat(q1, q2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1)
        # Decompose the Essential matrix into R and t
        R, t = self.decomp_essential_mat(E, q1, q2)
        ## use recover pose
        #points, R, t, mask_ = cv2.recoverPose(E, q1, q2, self.K)
        #print("det(R)", np.linalg.det(R))
        if np.linalg.det(R) < 0:
           print("det(R) < 0")

        # Get transformation matrix
        transformation_matrix = self._form_transf(R, np.squeeze(t))
        return transformation_matrix

    def decomp_essential_mat(self, E, q1, q2):
        """
        Decompose the Essential matrix

        Parameters
        ----------
        E (ndarray): Essential matrix
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image

        Returns
        -------
        right_pair (list): Contains the rotation matrix and translation vector
        """
        def sum_z_cal_relative_scale(R, t):
            # Get the transformation matrix
            T = self._form_transf(R, t)
            # Make the projection matrix
            P = np.matmul(np.concatenate((self.K, np.zeros((3, 1))), axis=1), T)

            # Triangulate the 3D points
            hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)
            # Also seen from cam 2
            hom_Q2 = np.matmul(T, hom_Q1)

            # Un-homogenize
            uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

            # Find the number of points there has positive z coordinate in both cameras
            sum_of_pos_z_Q1 = sum(uhom_Q1[2, :] > 0)
            sum_of_pos_z_Q2 = sum(uhom_Q2[2, :] > 0)

            # Form point pairs and calculate the relative scale
            relative_scale = np.mean(np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1)/
                                     np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1))
            return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale

        # Decompose the essential matrix
        R1, R2, t = cv2.decomposeEssentialMat(E)
        t = np.squeeze(t)

        # Make a list of the different possible pairs
        pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]

        # Check which solution there is the right one
        z_sums = []
        relative_scales = []
        for R, t in pairs:
            z_sum, scale = sum_z_cal_relative_scale(R, t)
            z_sums.append(z_sum)
            relative_scales.append(scale)

        # Select the pair there has the most points with positive z coordinate
        right_pair_idx = np.argmax(z_sums)
        right_pair = pairs[right_pair_idx]
        relative_scale = relative_scales[right_pair_idx]
        R1, t = right_pair
        

        return [R1, t]
