import os
import numpy as np
import cv2

from visualization import plotting
from visualization.video import play_trip

from tqdm import tqdm
import csv
import time

from scipy.spatial.transform import Rotation as Rot
from MonocularVisualOdometry import VisualOdometry


def main():

    Translation = np.array([0.00, 0.00, 0.00])
    Rotation = np.array([[1, 0, 0],[ 0, 1, 0], [0, 0, 1]])
    body_to_cam_angles = [-90.000, -90.0, 0.000]
    K = np.array([[692.881445252052,0,597.333157869075],
                    [0,686.823830607895,359.981351649336],
                    [0,0,1]])
    P = np.zeros((3, 4))
    P = np.matmul(K, np.hstack((Rotation, Translation.reshape(3, 1))))
    poses_timestamps = 'data/novatel-reference-postprocessed-inspva.csv'

    name = "oak"
    # speed sources are gt 
    speed_source = "gt"
    # dirctory for the image files
    data_dir = "data/images"
    # file path to the ground truth poses in txt format
    #poses_dir = "data/poses.txt" # if you only have the poses matrices you can use it but you need to provide timestamps in a csv file
    poses_dir = poses_timestamps 
    downsample = 1 # 1 for no downsampling, 2 for half the frequency, 3 for 1/3 of the frequency
    ## oak 30 , stereo 15, gps 50, odometer 16
    images_freq = 30
    gps_freq = 50
    delta_time = 0.0333 * downsample # 15 fps 0.066 , 30 fps 0.0333
    start = 0 #(start and end are indices of the images not timestamps)
    end = start + 10 * int(images_freq)

    show_trip = False
    visualize_matches = False
    

    ## loop to run all combinations of feature detectors and matchers
    feature_detector = "disk"  # choose one of "sift", "orb", "disk", "superpoint"
    feature_matcher = "snn"  # choose one of "flann", "snn","smnn", "fginn", "adalam", "lightglue"
    start = 0 # start from the beginning
    end = start + 10 * int(images_freq)

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "results_"+str(start)+"_"+str(images_freq/downsample)+"_"+speed_source+".csv"), "w") as file:
        writer = csv.writer(file)
        writer.writerow(["feature_detector", "feature_matcher", "processing_speed(pose/sec)", "time_under_1m(sec)", "max_error(m)","rmse(m)", "distance_traveled(m)", "drift_percentage"])

 
    # orb and lightglue are not compatible 
    if (feature_detector == "orb" and feature_matcher == "lightglue"):
        print("orb and lightglue are not compatible")
        return
    
    # initialize the visual odometry object
    vo = VisualOdometry(data_dir, poses_dir,poses_timestamps, start, end, downsample, images_freq, gps_freq, feature_detector, feature_matcher, K=K, P=P, visualization=visualize_matches, body_to_cam_angles=body_to_cam_angles)
   
    if show_trip:
        images = []
        for i in range(len(vo.image_paths)):
            images.append(cv2.imread(vo.image_paths[i], cv2.IMREAD_GRAYSCALE))  
        play_trip(images, waite_time=33)  

    gt_path = []
    estimated_path = []
    distance_traveled = 0
    start_time = time.time()
    print("len(vo.gt_poses)", len(vo.gt_poses))
    for i, gt_pose in enumerate(tqdm(vo.gt_poses, unit="pose")):

        if i == 0:
            cur_pose = gt_pose.copy()  
            rotation = Rot.from_matrix(cur_pose[:3,:3])
            euler_angles = rotation.as_euler('ZXY', degrees=True)
            print("euler_angles", euler_angles)
        else:
            q1, q2 = vo.get_matches(i)
            transf = vo.get_pose(q1, q2)

            distance_scale = np.sqrt((gt_pose[0,3]- vo.gt_poses[i-1][0,3])**2 + (gt_pose[1,3]-vo.gt_poses[i-1][1,3])**2 + (gt_pose[2,3]- vo.gt_poses[i-1][2,3])**2) 
            distance_traveled += distance_scale

            
            transf[:3,3] = transf[:3,3] * distance_scale

            cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))

        gt_path.append((gt_pose[0, 3], gt_pose[1, 3]))
        #print( gt_pose[0, 3], gt_pose[1, 3])
        estimated_path.append((cur_pose[0, 3], cur_pose[1, 3]))

    
    ## saving the results to the csv file
    end_time = time.time()
    total_time = end_time - start_time
    average_processing_speed = len(vo.gt_poses) / total_time
    print("average_processing_speed ", average_processing_speed, " pose/sec")
    # Writing to csv file
    with open(os.path.join(output_dir,"poses_"+feature_detector+"_"+feature_matcher+"_"+str(start)+"_"+str(images_freq/downsample)+"_"+speed_source+".csv"), mode='w', newline='') as file:
        writer = csv.writer(file) 
        writer.writerow(['ground_truth_x','ground_truth_y', 'Estimated_poses_x', 'Estimated_poses_y'])
        # write gt and estimated poses as float
        for gt, est in zip(gt_path, estimated_path):
            writer.writerow([gt[0], gt[1], est[0], est[1]])


    frame_num_more_than_1m, max_error, rmse, final_error = plotting.visualize_paths(gt_path, estimated_path, title="Visual Odometry : average_processing_speed "+ str(average_processing_speed) + "pose/sec", file_out=os.path.join(output_dir,os.path.basename(name) + "_"+feature_detector+"_"+feature_matcher+"_"+str(start)+"_"+str(images_freq/downsample)+"_"+speed_source+".html"))
    print("frame_num_more_than_1m", frame_num_more_than_1m)
    ## save the results to the csv file 
    with open(os.path.join(output_dir,"results_"+str(start)+"_"+str(images_freq/downsample)+"_"+speed_source+".csv"), "a") as file:
        writer = csv.writer(file)
        writer.writerow([feature_detector, feature_matcher, average_processing_speed, (frame_num_more_than_1m-1)/(images_freq/downsample), max_error, rmse, distance_traveled, (final_error/distance_traveled)*100])

            
if __name__ == "__main__":
    main()
