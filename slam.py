#!/usr/bin/env python3
"""
LiDAR-Odometry SLAM with Multiple Modes
==========================================
Dataset: SubSurfaceGeoRobo - 80m traverse

Modes:
  baseline              - LiDAR + odom (original working SLAM)
  standard_degraded     - Standard degradation (fog/rain): mild spurious
  heavy_degraded        - Heavy degradation (dust/snow): spurious + sector blank
  radar                 - LiDAR + radar fusion (30% radar weight)
  standard_degraded_radar  - Standard degraded + radar (70% radar weight)
  heavy_degraded_radar     - Heavy degraded + radar (90% radar weight)
  radar_only            - Radar-only pose estimation, LiDAR mapping
  odom_only             - Pure odometry (no SLAM correction)

Usage:
  python slam.py --bag <bag> --mode baseline
  python slam.py --bag <bag> --mode standard_degraded heavy_degraded
  python slam.py --bag <bag> --mode standard_degraded_radar heavy_degraded_radar
  python slam.py --bag <bag> --mode all
"""

import argparse
import numpy as np
from collections import deque
import time
import os
import copy

import rosbag
import sensor_msgs.point_cloud2 as pc2
from tf.transformations import (quaternion_matrix, quaternion_from_matrix,
                                 euler_from_quaternion)

import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d


ALL_MODES = ['baseline', 'standard_degraded', 'heavy_degraded',
             'radar', 'standard_degraded_radar', 'heavy_degraded_radar',
             'radar_only', 'odom_only']


###############################################################################
# 1. MATH
###############################################################################

def quat_to_rot(q):
    return quaternion_matrix(q)[:3, :3]

def rot_to_quat(R):
    mat = np.eye(4); mat[:3, :3] = R
    return quaternion_from_matrix(mat)

def yaw_to_rot(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def rot_to_yaw(R):
    return np.arctan2(R[1, 0], R[0, 0])

def pose_to_T(x, y, z, yaw):
    T = np.eye(4)
    T[:3, :3] = yaw_to_rot(yaw)
    T[:3, 3] = [x, y, z]
    return T


###############################################################################
# 2. LIDAR -> BASE TRANSFORM (Translation Only)
###############################################################################

def get_lidar_to_base():
    T = np.eye(4)
    T[0, 3] = 0.565
    T[1, 3] = -0.295
    T[2, 3] = 0.4585
    return T


def get_radar_to_base():
    """Identity rotation, translation only.
    Radar driver outputs points in corrected frame.
    """
    T_r1 = np.eye(4)
    T_r1[:3, 3] = [0.565, 0.295, 0.3785]

    T_r2 = np.eye(4)
    T_r2[:3, 3] = [-0.565, -0.295, 0.3785]

    print(f"[CALIB] Radar1->base t: {T_r1[:3, 3]} (identity)")
    print(f"[CALIB] Radar2->base t: {T_r2[:3, 3]} (identity)")
    
    return T_r1, T_r2


###############################################################################
# 3. DATA EXTRACTION
###############################################################################

def extract_data(bag_path, max_lidar_scans=None, read_radar=False):
    """Read LiDAR, odom, and optionally radar from bag."""
    print(f"[DATA] Reading {bag_path}...")
    bag = rosbag.Bag(bag_path)

    topics = ['/robot/front_laser/points', '/robot/robotnik_base_control/odom']
    if read_radar:
        topics += ['/isdr_driver_1/points', '/isdr_driver_2/points']

    lidar_data = []
    odom_data = []
    radar1_data = []
    radar2_data = []
    
    # Track raw message counts for the printout
    msg_counts = {t: 0 for t in topics}
    total_processed = 0

    for topic, msg, t in bag.read_messages(topics=topics):
        msg_counts[topic] += 1
        total_processed += 1
        stamp = msg.header.stamp.to_sec()

        # --- Heartbeat print so it doesn't look frozen ---
        if total_processed % 2000 == 0:
            print(f"  ... processed {total_processed} messages ...")

        if topic == '/robot/front_laser/points':
            if max_lidar_scans and len(lidar_data) >= max_lidar_scans:
                continue
            # FAST PARSING: list() pushes the loop down to C level
            pts_gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            points = np.array(list(pts_gen), dtype=np.float32)
            
            if len(points) > 100:
                lidar_data.append((stamp, points))

        elif topic == '/robot/robotnik_base_control/odom':
            pos = msg.pose.pose.position
            ori = msg.pose.pose.orientation
            q = [ori.x, ori.y, ori.z, ori.w]
            _, _, yaw = euler_from_quaternion(q)
            vx = msg.twist.twist.linear.x
            wz = msg.twist.twist.angular.z
            odom_data.append((stamp, pos.x, pos.y, pos.z, yaw, vx, wz))

        elif topic == '/isdr_driver_1/points':
            pts_gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            points = np.array(list(pts_gen), dtype=np.float32)
            if len(points) > 0:
                radar1_data.append((stamp, points))

        elif topic == '/isdr_driver_2/points':
            pts_gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            points = np.array(list(pts_gen), dtype=np.float32)
            if len(points) > 0:
                radar2_data.append((stamp, points))

    bag.close()
    
    lidar_data.sort(key=lambda x: x[0])
    odom_data.sort(key=lambda x: x[0])
    radar1_data.sort(key=lambda x: x[0])
    radar2_data.sort(key=lambda x: x[0])

    print(f"\n[DATA] Extraction Complete:")
    print(f"  --> LiDAR  (/robot/front_laser/points)         : read {msg_counts['/robot/front_laser/points']} msgs, kept {len(lidar_data)} valid scans")
    print(f"  --> Odom   (/robot/robotnik_base_control/odom) : read {msg_counts['/robot/robotnik_base_control/odom']} msgs, kept {len(odom_data)} poses")
    if read_radar:
        print(f"  --> Radar1 (/isdr_driver_1/points)             : read {msg_counts['/isdr_driver_1/points']} msgs, kept {len(radar1_data)} valid points")
        print(f"  --> Radar2 (/isdr_driver_2/points)             : read {msg_counts['/isdr_driver_2/points']} msgs, kept {len(radar2_data)} valid points")
    print("-" * 60)

    return lidar_data, odom_data, radar1_data, radar2_data


###############################################################################
# 4. ODOM INTERPOLATOR
###############################################################################

class OdomInterpolator:
    def __init__(self, odom_data):
        t = np.array([d[0] for d in odom_data])
        x = np.array([d[1] for d in odom_data])
        y = np.array([d[2] for d in odom_data])
        z = np.array([d[3] for d in odom_data])
        yaw = np.array([d[4] for d in odom_data])
        vx = np.array([d[5] for d in odom_data])
        yaw_unwrapped = np.unwrap(yaw)

        self.f_x = interp1d(t, x, fill_value='extrapolate')
        self.f_y = interp1d(t, y, fill_value='extrapolate')
        self.f_z = interp1d(t, z, fill_value='extrapolate')
        self.f_yaw = interp1d(t, yaw_unwrapped, fill_value='extrapolate')
        self.f_vx = interp1d(t, vx, fill_value='extrapolate')
        self.x0, self.y0, self.z0 = x[0], y[0], z[0]
        self.yaw0 = yaw_unwrapped[0]

    def get_pose(self, t):
        dx = float(self.f_x(t)) - self.x0
        dy = float(self.f_y(t)) - self.y0
        dz = float(self.f_z(t)) - self.z0
        yaw = float(self.f_yaw(t)) - self.yaw0
        c, s = np.cos(-self.yaw0), np.sin(-self.yaw0)
        x = c * dx - s * dy
        y = s * dx + c * dy
        return x, y, dz, yaw

    def get_relative_transform(self, t1, t2):
        x1, y1, z1, yaw1 = self.get_pose(t1)
        x2, y2, z2, yaw2 = self.get_pose(t2)
        T1 = pose_to_T(x1, y1, z1, yaw1)
        T2 = pose_to_T(x2, y2, z2, yaw2)
        return np.linalg.inv(T1) @ T2


###############################################################################
# 5. LIDAR DEGRADATION SIMULATOR
###############################################################################

class LiDARDegrader:
    """Physically realistic LiDAR degradation for dust/fog/snow.
    
    Real-world effects simulated:
      1. Range attenuation: signal absorbed by particles → reduced max range
      2. Spurious returns: backscatter from dust/fog/snow creates phantom points
         at random ranges along existing beam directions
      3. Range noise: partial returns shift measured range (multi-path)
      4. Signal dropout: some beams fully absorbed → missing points
      5. Sector occlusion (heavy only): snow/mud on sensor window
    
    Presets:
      'standard' - Standard fog/rain: mild spurious, moderate range reduction
      'heavy'    - Heavy dust/snow: lots of spurious, severe range cut, sector blank
    """
    def __init__(self, severity='standard'):
        self.severity = severity
        if severity == 'standard':
            self.max_range = np.inf
            self.dropout_rate = 0.08
            self.range_noise_std = 0.02
            self.spurious_ratio = 0.02     # 2%
            self.spurious_max_range = 3.0
            self.spurious_cluster_std = 0.1
            self.sector_dropout = False
            self.blank_yaw_width = 0
        elif severity == 'heavy':
            self.max_range = np.inf
            self.dropout_rate = 0.25
            self.range_noise_std = 0.05
            self.spurious_ratio = 0.08     # 8%
            self.spurious_max_range = 3.0
            self.spurious_cluster_std = 0.2
            self.sector_dropout = False
            self.blank_yaw_width = 0

        print(f"[DEGRADE] {severity.upper()} degradation:")
        print(f"  Dropout: {self.dropout_rate*100:.0f}% | "
              f"Range noise: {self.range_noise_std:.2f}m")
        print(f"  Spurious ratio: {self.spurious_ratio*100:.0f}% | "
              f"Spurious range: {self.spurious_max_range:.0f}m")
        if self.sector_dropout:
            print(f"  Sector blanked: ±{np.degrees(self.blank_yaw_width/2):.0f}°")

    def degrade(self, points):
        if len(points) == 0:
            return points

        # 1. Range attenuation
        ranges = np.linalg.norm(points, axis=1)
        mask = ranges < self.max_range
        points = points[mask]
        ranges = ranges[mask]

        if len(points) == 0:
            return points

        # 2. Sector occlusion (heavy only)
        if self.sector_dropout and len(points) > 0:
            yaws = np.arctan2(points[:, 1], points[:, 0])
            angle_diff = np.abs(np.arctan2(
                np.sin(yaws - self.blank_yaw_center),
                np.cos(yaws - self.blank_yaw_center)))
            keep = angle_diff > (self.blank_yaw_width / 2)
            points = points[keep]
            ranges = ranges[keep]

        if len(points) == 0:
            return points

        # 3. Random dropout (signal absorption)
        keep = np.random.random(len(points)) > self.dropout_rate
        points = points[keep]
        ranges = ranges[keep]

        if len(points) == 0:
            return points

        # 4. Range noise (partial returns, multi-path)
        # Noise is along the beam direction, not isotropic
        directions = points / (ranges[:, None] + 1e-6)
        range_jitter = np.random.normal(0, self.range_noise_std, len(points))
        points = points + directions * range_jitter[:, None]

        # 5. Spurious returns (backscatter from particles)
        # Random scatter around robot — NOT along beam directions
        # so ICP can't mistake them for rotated wall structure
        n_spurious = int(len(points) * self.spurious_ratio)
        if n_spurious > 0:
            # Random angles and ranges
            angles = np.random.uniform(0, 2 * np.pi, n_spurious)
            spur_ranges = np.random.uniform(0.5, self.spurious_max_range, n_spurious)
            spur_pts = np.zeros((n_spurious, 3))
            spur_pts[:, 0] = spur_ranges * np.cos(angles)
            spur_pts[:, 1] = spur_ranges * np.sin(angles)
            spur_pts[:, 2] = np.random.normal(0, self.spurious_cluster_std, n_spurious)
            points = np.vstack([points, spur_pts])

        return points


###############################################################################
# 6. LIDAR PREPROCESSOR
###############################################################################

class LiDARPreprocessor:
    def __init__(self, voxel_size=0.15, min_range=1.0, max_range=50.0):
        self.voxel_size = voxel_size
        self.min_range = min_range
        self.max_range = max_range

    def process(self, points):
        ranges = np.linalg.norm(points, axis=1)
        mask = (ranges > self.min_range) & (ranges < self.max_range)
        points = points[mask]
        
        pcd = o3d.geometry.PointCloud()
        if len(points) < 50:
            pcd.points = o3d.utility.Vector3dVector(points if len(points) > 0 else np.zeros((1, 3)))
            return pcd
            
        pcd.points = o3d.utility.Vector3dVector(points)
        if len(points) > 500:
            pcd = pcd.voxel_down_sample(self.voxel_size)
            
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.voxel_size * 4, max_nn=30))
        return pcd


###############################################################################
# 7. LOCAL SUBMAP
###############################################################################

class LocalSubmap:
    def __init__(self, max_scans=20, voxel_size=0.25):
        self.max_scans = max_scans
        self.voxel_size = voxel_size
        self.scan_queue = deque()
        self.combined = None

    def add_scan(self, pcd_body, T_world):
        pcd_world = copy.deepcopy(pcd_body).transform(T_world)
        self.scan_queue.append(pcd_world)
        while len(self.scan_queue) > self.max_scans:
            self.scan_queue.popleft()
        self._rebuild()

    def _rebuild(self):
        combined = o3d.geometry.PointCloud()
        for pcd in self.scan_queue:
            combined += pcd
        combined = combined.voxel_down_sample(self.voxel_size)
        combined.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.voxel_size * 4, max_nn=30))
        self.combined = combined

    def get(self):
        return self.combined


###############################################################################
# 8. SCAN MATCHER
###############################################################################

class ScanMatcher:
    def match(self, source, target, init_T=np.eye(4)):
        T, f, r = self._icp(source, target, init_T, 1.0, 50)
        if f > 0.3:
            return T, f, r
            
        T2, f2, r2 = self._icp(source, target, init_T, 3.0, 30)
        T3, f3, r3 = self._icp_p2p(source, target, T2 if f2 > f else init_T, 2.0, 40)
        
        best = max([(f, T, r), (f2, T2, r2), (f3, T3, r3)], key=lambda item: item[0])
        return best[1], best[0], best[2]

    def _icp(self, src, tgt, init_T, max_d, max_i):
        r = o3d.pipelines.registration.registration_icp(
            src, tgt, max_d, init_T,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_i, relative_fitness=1e-6, relative_rmse=1e-6))
        return r.transformation, r.fitness, r.inlier_rmse
        
    def _icp_p2p(self, src, tgt, init_T, max_d, max_i):
        r = o3d.pipelines.registration.registration_icp(
            src, tgt, max_d, init_T,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_i, relative_fitness=1e-6, relative_rmse=1e-6))
        return r.transformation, r.fitness, r.inlier_rmse


###############################################################################
# 9. RADAR HELPER
###############################################################################

def find_nearest_radar(t_scan, radar_data, t_window=4.0):
    """Accumulate all individual radar points within +/- t_window seconds.
    Returns list of (timestamp, points) tuples for motion compensation.
    Uses binary search for efficiency since data is sorted."""
    if not radar_data:
        return None

    lo, hi = 0, len(radar_data) - 1
    t_lo = t_scan - t_window
    t_hi = t_scan + t_window

    while lo < hi:
        mid = (lo + hi) // 2
        if radar_data[mid][0] < t_lo:
            lo = mid + 1
        else:
            hi = mid

    accumulated = []
    for i in range(lo, len(radar_data)):
        t_r = radar_data[i][0]
        if t_r > t_hi:
            break
        accumulated.append((t_r, radar_data[i][1]))

    if not accumulated:
        return None
    return accumulated


def accumulate_radar_compensated(radar_hits, t_scan, odom, T_radar_to_base,
                                  min_range=0.5, max_range=5.0):
    """Motion-compensate radar points to the pose at t_scan.
    Filters by range in radar frame before transforming.
    """
    R_r2b = T_radar_to_base[:3, :3]
    t_r2b = T_radar_to_base[:3, 3]

    compensated = []
    for t_r, pts in radar_hits:
        # Range filter in radar frame
        ranges = np.linalg.norm(pts, axis=1)
        mask = (ranges >= min_range) & (ranges <= max_range)
        pts_filt = pts[mask]
        if len(pts_filt) == 0:
            continue

        # Transform to base frame
        pts_base = (R_r2b @ pts_filt.T).T + t_r2b
        # Motion compensation: from pose@t_r to pose@t_scan
        x1, y1, z1, yaw1 = odom.get_pose(t_r)
        x2, y2, z2, yaw2 = odom.get_pose(t_scan)
        T1 = pose_to_T(x1, y1, z1, yaw1)
        T2 = pose_to_T(x2, y2, z2, yaw2)
        T_rel = np.linalg.inv(T2) @ T1
        pts_comp = (T_rel[:3, :3] @ pts_base.T).T + T_rel[:3, 3]
        compensated.append(pts_comp)

    return np.vstack(compensated) if compensated else None

def transform_radar_to_base(pts, T_radar_to_base):
    R = T_radar_to_base[:3, :3]
    t = T_radar_to_base[:3, 3]
    return (R @ pts.T).T + t

def make_radar_pcd(pts, voxel_size=0.3):
    if pts is None or len(pts) < 5:
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd = pcd.voxel_down_sample(voxel_size)
    if len(pcd.points) > 10:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size * 4, max_nn=30))
    return pcd



###############################################################################
# 9b. RADAR SUBMAP, ICP, AND SCAN BUILDER (matches radar_slam.py params)
###############################################################################

class _RadarOnlySubmap:
    def __init__(self, max_scans=10, voxel_size=0.15):
        self.max_scans = max_scans
        self.voxel_size = voxel_size
        self.scan_queue = deque()
        self.combined = None

    def add_scan(self, pts_base, T_world):
        if pts_base is None or len(pts_base) < 3:
            return
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_base)
        pcd_world = pcd.transform(T_world)
        self.scan_queue.append(pcd_world)
        while len(self.scan_queue) > self.max_scans:
            self.scan_queue.popleft()
        self._rebuild()

    def _rebuild(self):
        combined = o3d.geometry.PointCloud()
        for pcd in self.scan_queue:
            combined += pcd
        if len(combined.points) > 10:
            combined = combined.voxel_down_sample(self.voxel_size)
        self.combined = combined

    def get(self):
        return self.combined

    def size(self):
        return len(self.combined.points) if self.combined else 0


def _filter_statistical(pts, nb_neighbors=8, std_ratio=1.5):
    """Statistical + radius outlier removal (same as radar_slam.py)."""
    if pts is None or len(pts) < 20:
        return pts
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    filtered, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    if len(filtered.points) > 20:
        filtered, _ = filtered.remove_radius_outlier(nb_points=5, radius=1.0)
    if len(filtered.points) < 5:
        return pts
    return np.asarray(filtered.points)


def _build_radar_scan(t_scan, radar1_data, radar2_data, odom,
                      T_r1_to_base, T_r2_to_base,
                      t_window=4.0, min_range=0.5, max_range=5.0):
    """Build one filtered, motion-compensated radar scan from both radars."""
    all_pts = []
    for rdata, T_r in [(radar1_data, T_r1_to_base), (radar2_data, T_r2_to_base)]:
        hits = find_nearest_radar(t_scan, rdata, t_window)
        if hits:
            pts = accumulate_radar_compensated(hits, t_scan, odom, T_r,
                                               min_range, max_range)
            if pts is not None:
                all_pts.append(pts)
    if all_pts:
        combined = np.vstack(all_pts)
        return _filter_statistical(combined)
    return None


def _radar_icp(radar_pts, submap, T_init, max_dist=0.5):
    """Multi-resolution ICP for sparse radar (same as radar_slam.py).
    Coarse pass then fine pass. Returns (transform, fitness) or (None, 0.0)."""
    target = submap.get()
    if target is None or len(target.points) < 30:
        return None, 0.0

    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(radar_pts)
    if len(src.points) < 15:
        return None, 0.0

    try:
        # Coarse
        r1 = o3d.pipelines.registration.registration_icp(
            src, target, max_dist, T_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=30, relative_fitness=1e-6, relative_rmse=1e-6))

        # Fine
        r2 = o3d.pipelines.registration.registration_icp(
            src, target, max_dist * 0.5, r1.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=30, relative_fitness=1e-6, relative_rmse=1e-6))

        n_inliers = int(r2.fitness * len(src.points))
        if n_inliers < 10 or r2.fitness < 0.2:
            return None, r2.fitness

        return r2.transformation, r2.fitness
    except:
        return None, 0.0


###############################################################################
# 10. SLAM LOOP
###############################################################################

def run_slam(lidar_data, odom_data, T_lidar_to_base,
             radar1_data=None, radar2_data=None,
             mode='baseline', submap_size=20, map_interval=5):
             
    np.random.seed(42) 
    
    odom = OdomInterpolator(odom_data)
    preprocessor = LiDARPreprocessor(voxel_size=0.15)
    matcher = ScanMatcher()
    submap = LocalSubmap(max_scans=submap_size, voxel_size=0.25)

    R_l2b = T_lidar_to_base[:3, :3]
    t_l2b = T_lidar_to_base[:3, 3]

    degrader = None
    use_radar = False
    radar_weight = 0.3  # default for radar mode (no degradation)
    T_r1_to_base, T_r2_to_base = None, None
    radar_submap = None

    if mode in ('standard_degraded', 'standard_degraded_radar'):
        degrader = LiDARDegrader(severity='standard')
        radar_weight = 0.7
    elif mode in ('heavy_degraded', 'heavy_degraded_radar'):
        degrader = LiDARDegrader(severity='heavy')
        radar_weight = 0.9

    radar_only_mode = (mode == 'radar_only')

    if mode in ('radar', 'standard_degraded_radar', 'heavy_degraded_radar', 'radar_only'):
        use_radar = True
        T_r1_to_base, T_r2_to_base = get_radar_to_base()
        radar_submap = _RadarOnlySubmap(max_scans=10, voxel_size=0.15)
        if radar_only_mode:
            print(f"[RADAR] Radar-only pose estimation, LiDAR mapping")
        else:
            print(f"[RADAR] Separate radar constraint ON (weight={radar_weight})")

    x0, y0, z0, yaw0 = odom.get_pose(lidar_data[0][0])
    global_pose = pose_to_T(x0, y0, z0, yaw0)

    trajectory = []
    map_pcds = []
    fitness_log = []
    radar_fit_log = []
    weight_log = []
    n_scans = len(lidar_data)
    t0 = time.time()
    low_fit = 0
    radar_corrections = 0

    print(f"\n[SLAM] Mode: {mode} | {n_scans} scans | submap={submap_size}")

    for si in range(n_scans):
        t_scan = lidar_data[si][0]
        pts_raw = lidar_data[si][1]

        # --- Keep original for map building ---
        pts_raw_clean = pts_raw

        # --- Build radar scan (always, before degradation) ---
        radar_pts = None
        if use_radar:
            radar_pts = _build_radar_scan(
                t_scan, radar1_data, radar2_data, odom,
                T_r1_to_base, T_r2_to_base)

        # --- Degrade LiDAR (only affects ICP, not the map) ---
        if degrader is not None:
            pts_raw = degrader.degrade(pts_raw)
            if len(pts_raw) < 20:
                # LiDAR completely failed
                if si > 0:
                    t_prev = lidar_data[si - 1][0]
                    T_odom_rel = odom.get_relative_transform(t_prev, t_scan)
                    T_odom_pose = (trajectory[-1][1] @ T_odom_rel).copy()

                    # Try radar-only correction
                    if use_radar and radar_pts is not None and radar_submap.size() >= 40:
                        T_radar, r_fit = _radar_icp(radar_pts, radar_submap, T_odom_pose)
                        if T_radar is not None:
                            w = radar_weight
                            yaw_o = rot_to_yaw(T_odom_pose[:3, :3])
                            yaw_r = rot_to_yaw(T_radar[:3, :3])
                            dyaw = np.arctan2(np.sin(yaw_r - yaw_o),
                                              np.cos(yaw_r - yaw_o))
                            global_pose = np.eye(4)
                            global_pose[:3, :3] = yaw_to_rot(yaw_o + w * dyaw)
                            global_pose[:3, 3] = (1-w)*T_odom_pose[:3, 3] + w*T_radar[:3, 3]
                            radar_corrections += 1
                        else:
                            global_pose = T_odom_pose.copy()
                    else:
                        global_pose = T_odom_pose.copy()

                    global_pose[2, 3] *= 0.9
                    global_pose[:3, :3] = yaw_to_rot(rot_to_yaw(global_pose[:3, :3]))

                trajectory.append((t_scan, global_pose.copy()))
                fitness_log.append(0.0)
                radar_fit_log.append(0.0)
                weight_log.append(0.0)
                if use_radar and radar_pts is not None:
                    radar_submap.add_scan(radar_pts, global_pose)
                # Still add clean scan to map
                if si % map_interval == 0:
                    clean_base = (R_l2b @ pts_raw_clean.T).T + t_l2b
                    clean_pcd = preprocessor.process(clean_base)
                    map_pcds.append(copy.deepcopy(clean_pcd).transform(global_pose))
                continue

        # --- Build LiDAR point cloud ---
        pts_base = (R_l2b @ pts_raw.T).T + t_l2b
        cur_pcd = preprocessor.process(pts_base)

        if si == 0:
            trajectory.append((t_scan, global_pose.copy()))
            submap.add_scan(cur_pcd, global_pose)
            if use_radar and radar_pts is not None:
                radar_submap.add_scan(radar_pts, global_pose)
            # Map uses CLEAN (undegraded) scan
            clean_base = (R_l2b @ pts_raw_clean.T).T + t_l2b
            clean_pcd = preprocessor.process(clean_base)
            map_pcds.append(copy.deepcopy(clean_pcd).transform(global_pose))
            fitness_log.append(1.0)
            radar_fit_log.append(0.0)
            weight_log.append(0.0)
            n_r = len(radar_pts) if radar_pts is not None else 0
            print(f"  [Scan 0] Init. {len(cur_pcd.points)} lidar pts | {n_r} radar pts")
            continue

        t_prev = lidar_data[si - 1][0]
        T_odom_rel = odom.get_relative_transform(t_prev, t_scan)
        T_init = trajectory[-1][1] @ T_odom_rel

        # --- Radar-only mode: skip lidar ICP, use radar ICP + odom ---
        if radar_only_mode:
            fitness = 0.0
            rmse = 0.0
            r_fit = 0.0
            w_radar = 0.0
            T_result = T_init

            if radar_pts is not None and len(radar_pts) >= 15 and radar_submap.size() >= 50:
                T_icp, r_fit = _radar_icp(radar_pts, radar_submap, T_init, max_dist=0.5)
                if T_icp is not None and r_fit > 0.2:
                    # Adaptive blend: same as radar_slam.py
                    w = min(0.5, r_fit * 0.6)
                    yaw_odom = rot_to_yaw(T_init[:3, :3])
                    yaw_icp = rot_to_yaw(T_icp[:3, :3])
                    dyaw = np.arctan2(np.sin(yaw_icp - yaw_odom),
                                      np.cos(yaw_icp - yaw_odom))
                    T_result = np.eye(4)
                    T_result[:3, :3] = yaw_to_rot(yaw_odom + w * dyaw)
                    T_result[:3, 3] = (1 - w) * T_init[:3, 3] + w * T_icp[:3, 3]
                    radar_corrections += 1
                    w_radar = w

            global_pose = T_result.copy()
            global_pose[2, 3] = 0.0
            global_pose[:3, :3] = yaw_to_rot(rot_to_yaw(global_pose[:3, :3]))
            fitness_log.append(r_fit)
            radar_fit_log.append(r_fit)
            weight_log.append(w_radar)

        else:
            # --- LiDAR ICP: lidar scan vs lidar-only submap ---
            T_lidar, fitness, rmse = matcher.match(cur_pcd, submap.get(), T_init)
            lidar_ok = fitness >= 0.15
            if not lidar_ok:
                low_fit += 1
                T_lidar = T_init

            # --- Radar ICP: radar scan vs radar-only submap ---
            T_radar_result = None
            r_fit = 0.0
            w_radar = 0.0
            if use_radar and radar_pts is not None and radar_submap.size() >= 50:
                T_radar_icp, r_fit = _radar_icp(radar_pts, radar_submap, T_init)
                if T_radar_icp is not None:
                    T_radar_result = T_radar_icp
                    radar_corrections += 1

            # --- Fuse: blend LiDAR + radar pose estimates ---
            if T_radar_result is not None:
                w_radar = radar_weight

                yaw_l = rot_to_yaw(T_lidar[:3, :3])
                yaw_r = rot_to_yaw(T_radar_result[:3, :3])
                dyaw = np.arctan2(np.sin(yaw_r - yaw_l), np.cos(yaw_r - yaw_l))
                global_pose = np.eye(4)
                global_pose[:3, :3] = yaw_to_rot(yaw_l + w_radar * dyaw)
                global_pose[:3, 3] = (1-w_radar)*T_lidar[:3, 3] + w_radar*T_radar_result[:3, 3]
            else:
                global_pose = T_lidar.copy()

            global_pose[2, 3] *= 0.9
            global_pose[:3, :3] = yaw_to_rot(rot_to_yaw(global_pose[:3, :3]))
            fitness_log.append(fitness)
            radar_fit_log.append(r_fit)
            weight_log.append(w_radar)

        trajectory.append((t_scan, global_pose.copy()))
        submap.add_scan(cur_pcd, global_pose)
        if use_radar and radar_pts is not None:
            radar_submap.add_scan(radar_pts, global_pose)

        if si % map_interval == 0:
            # Map uses CLEAN (undegraded) scan
            clean_base = (R_l2b @ pts_raw_clean.T).T + t_l2b
            clean_pcd = preprocessor.process(clean_base)
            map_pcds.append(copy.deepcopy(clean_pcd).transform(global_pose))

        if si % 200 == 0:
            p = global_pose[:3, 3]
            if use_radar:
                r_str = (f"  r_fit={r_fit:.3f}  w_r={w_radar:.2f}  "
                         f"corr={radar_corrections}")
            else:
                r_str = ""
            print(f"  [Scan {si:4d}/{n_scans}] "
                  f"pos=({p[0]:7.2f}, {p[1]:7.2f}, {p[2]:7.2f})  "
                  f"l_fit={fitness:.3f}  rmse={rmse:.4f}"
                  f"{r_str}  t={time.time()-t0:.0f}s")

    total = time.time() - t0
    print(f"\n[SLAM] Done. {n_scans} scans in {total:.1f}s ({n_scans/total:.1f} Hz)")
    print(f"  Low fitness: {low_fit}/{n_scans}")
    if use_radar:
        print(f"  Radar corrections: {radar_corrections}/{n_scans}")
    return trajectory, map_pcds, fitness_log, radar_fit_log, weight_log



###############################################################################
# 11. SAVE
###############################################################################

def save_results(trajectory, map_pcds, fitness_log, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    traj_arr = []
    for t, T in trajectory:
        p = T[:3, 3]
        q = rot_to_quat(T[:3, :3])
        traj_arr.append([t, p[0], p[1], p[2], q[0], q[1], q[2], q[3]])
    traj_arr = np.array(traj_arr)
    np.savetxt(os.path.join(output_dir, "trajectory.txt"), traj_arr,
               header="timestamp x y z qx qy qz qw", fmt="%.6f")

    if map_pcds:
        combined = o3d.geometry.PointCloud()
        for pcd in map_pcds:
            combined += pcd
        combined = combined.voxel_down_sample(0.3)
        o3d.io.write_point_cloud(os.path.join(output_dir, "global_map.pcd"), combined)
        print(f"[SAVE] Map: {len(combined.points)} pts -> {output_dir}")

    np.savetxt(os.path.join(output_dir, "fitness_log.txt"),
               np.array(fitness_log), fmt="%.4f")

    print(f"[SAVE] Trajectory: {len(traj_arr)} poses -> {output_dir}")
    return traj_arr


###############################################################################
# 12. VISUALIZATION (per-mode)
###############################################################################

def visualize_mode(traj_arr, odom_data, output_dir, mode_name,
                   fitness_log=None, radar_fit_log=None, weight_log=None):
    slam_t = traj_arr[:, 0]
    slam_pos = traj_arr[:, 1:4]
    t_rel = slam_t - slam_t[0]

    odom_t = np.array([d[0] for d in odom_data])
    odom_raw = np.array([[d[1], d[2], d[3]] for d in odom_data])
    idx0 = np.argmin(np.abs(odom_t - slam_t[0]))
    odom_pos = odom_raw - odom_raw[idx0]
    odom_yaw0 = odom_data[idx0][4]
    c, s = np.cos(-odom_yaw0), np.sin(-odom_yaw0)
    R2d = np.array([[c, -s], [s, c]])
    odom_pos[:, :2] = (R2d @ odom_pos[:, :2].T).T

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax = axes[0]
    ax.plot(slam_pos[:, 0], slam_pos[:, 1], 'b-', lw=2, label='SLAM')
    ax.plot(odom_pos[:, 0], odom_pos[:, 1], 'r--', lw=1.5, alpha=0.6, label='Odom')
    ax.plot(slam_pos[0, 0], slam_pos[0, 1], 'go', ms=10, label='Start')
    ax.plot(slam_pos[-1, 0], slam_pos[-1, 1], 'rs', ms=10, label='End')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_title(f'{mode_name} — Trajectory')
    ax.legend(); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t_rel, slam_pos[:, 2], 'b-', lw=2)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Z (m)')
    ax.set_title(f'{mode_name} — Height Profile')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "trajectory_2d.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # 3D
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(slam_pos[:, 0], slam_pos[:, 1], slam_pos[:, 2], 'b-', lw=2, label='SLAM')
    ax.plot(odom_pos[:, 0], odom_pos[:, 1], odom_pos[:, 2], 'r--', lw=1.5, alpha=0.6, label='Odom')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(f'{mode_name} — 3D Trajectory'); ax.legend()
    plt.savefig(os.path.join(output_dir, "trajectory_3d.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Fusion diagnostics (only for modes with radar data)
    has_radar = (radar_fit_log is not None and len(radar_fit_log) > 0
                 and any(r > 0 for r in radar_fit_log))
    if has_radar:
        r_fit = np.array(radar_fit_log[:len(t_rel)])
        w_rad = np.array(weight_log[:len(t_rel)])
        l_fit = np.array(fitness_log[:len(t_rel)]) if fitness_log else np.zeros(len(t_rel))

        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        ax = axes[0]
        ax.plot(t_rel, l_fit, 'b-', lw=1, alpha=0.7, label='LiDAR fitness')
        window = min(20, len(l_fit) // 10 + 1)
        if window > 1:
            smooth = np.convolve(l_fit, np.ones(window)/window, mode='same')
            ax.plot(t_rel, smooth, 'b-', lw=2, alpha=0.9, label=f'LiDAR (avg {window})')
        ax.set_ylabel('LiDAR Fitness')
        ax.set_ylim(0, 1.05)
        ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(t_rel, r_fit, 'g-', lw=1, alpha=0.7, label='Radar fitness')
        if window > 1:
            smooth_r = np.convolve(r_fit, np.ones(window)/window, mode='same')
            ax.plot(t_rel, smooth_r, 'g-', lw=2, alpha=0.9, label=f'Radar (avg {window})')
        ax.set_ylabel('Radar Fitness')
        ax.set_ylim(0, 1.05)
        ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[2]
        ax.plot(t_rel, w_rad, 'r-', lw=1, alpha=0.7, label='w_radar')
        if window > 1:
            smooth_w = np.convolve(w_rad, np.ones(window)/window, mode='same')
            ax.plot(t_rel, smooth_w, 'r-', lw=2, alpha=0.9, label=f'w_radar (avg {window})')
        ax.set_ylabel('Radar Weight')
        ax.set_ylim(0, 0.65)
        ax.set_xlabel('Time (s)')
        ax.legend(); ax.grid(True, alpha=0.3)

        plt.suptitle(f'{mode_name} — Fusion Diagnostics', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "fusion_diagnostics.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[VIS] Saved fusion_diagnostics.png to {output_dir}")

    print(f"[VIS] Saved plots to {output_dir}")


###############################################################################
# 13. ENTRY POINT
###############################################################################

def run_one_mode(mode, lidar_data, odom_data, radar1_data, radar2_data,
                 T_l2b, args):
    output_dir = os.path.join(args.output, mode)
    print(f"\n{'='*60}")
    print(f"  MODE: {mode}")
    print(f"{'='*60}")

    if mode == 'odom_only':
        trajectory, map_pcds, fitness_log, r_fit_log, w_log = run_odom_only(
            lidar_data, odom_data, T_l2b)
    else:
        trajectory, map_pcds, fitness_log, r_fit_log, w_log = run_slam(
            lidar_data, odom_data, T_l2b,
            radar1_data=radar1_data, radar2_data=radar2_data,
            mode=mode)

    traj_arr = save_results(trajectory, map_pcds, fitness_log, output_dir)
    visualize_mode(traj_arr, odom_data, output_dir, mode,
                   fitness_log=fitness_log,
                   radar_fit_log=r_fit_log, weight_log=w_log)

    d = np.linalg.norm(traj_arr[-1, 1:4] - traj_arr[0, 1:4])
    avg_fitness = np.mean(fitness_log) if fitness_log else 0
    print(f"[{mode}] Displacement: {d:.2f}m | Avg fitness: {avg_fitness:.3f}")
    return traj_arr


def run_odom_only(lidar_data, odom_data, T_lidar_to_base, map_interval=5):
    """Pure odometry — no SLAM correction. Baseline for comparison."""
    odom = OdomInterpolator(odom_data)
    preprocessor = LiDARPreprocessor(voxel_size=0.15)
    R_l2b = T_lidar_to_base[:3, :3]
    t_l2b = T_lidar_to_base[:3, 3]

    trajectory = []
    map_pcds = []
    fitness_log = []
    n_scans = len(lidar_data)

    print(f"\n[ODOM ONLY] {n_scans} scans — pure odometry, no ICP")
    t0 = time.time()

    for si in range(n_scans):
        t_scan = lidar_data[si][0]
        x, y, z, yaw = odom.get_pose(t_scan)
        T_world = pose_to_T(x, y, z, yaw)

        trajectory.append((t_scan, T_world.copy()))
        fitness_log.append(0.0)

        if si % map_interval == 0:
            pts_raw = lidar_data[si][1]
            pts_base = (R_l2b @ pts_raw.T).T + t_l2b
            cur_pcd = preprocessor.process(pts_base)
            map_pcds.append(copy.deepcopy(cur_pcd).transform(T_world))

    print(f"[ODOM ONLY] Done. {time.time()-t0:.1f}s")
    return trajectory, map_pcds, fitness_log, [], []


def main():
    parser = argparse.ArgumentParser(description="Multi-mode SLAM")
    parser.add_argument("--bag", required=True)
    parser.add_argument("--mode", nargs='+', default=["baseline"],
                        help="One or more modes, or 'all' (e.g. --mode degraded degraded_radar)")
    parser.add_argument("--output", default="slam_output")
    args = parser.parse_args()

    # Resolve modes
    if 'all' in args.mode:
        modes = ALL_MODES
    else:
        modes = args.mode
        for m in modes:
            if m not in ALL_MODES:
                print(f"Unknown mode: {m}. Choose from: {ALL_MODES}")
                return

    need_radar = any(m in ('radar', 'standard_degraded_radar', 'heavy_degraded_radar',
                          'radar_only')
                     for m in modes)

    T_l2b = get_lidar_to_base()
    lidar_data, odom_data, radar1_data, radar2_data = extract_data(
        args.bag, read_radar=need_radar)

    for mode in modes:
        run_one_mode(mode, lidar_data, odom_data,
                     radar1_data, radar2_data, T_l2b, args)

    print(f"\n[DONE] Results in {args.output}/")
    for m in modes:
        print(f"  {args.output}/{m}/")


if __name__ == "__main__":
    main()