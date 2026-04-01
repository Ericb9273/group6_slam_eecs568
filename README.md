slam.py reads in the rosbag dataset from location you specify and saves the output to slam_output folder

Usage: in this folder, run `python3 slam.py --bag ~/datasets/03_80m_other_sensor.bag` , change bag file path to your file path

To visualize the SLAM map, run `python3 view_map.py`

To compare with ground truth, run `python3 compare_gt.py --gt ~/datasets/3D_point_cloud_GT.las --slam slam_output/global_map.pcd --view`
