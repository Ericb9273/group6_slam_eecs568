#!/usr/bin/env python3
"""
Radar-Only SLAM
===============
Pure radar SLAM with identity transforms, outlier filtering,
and tuned ICP for sparse 2D radar data.

Outputs:
  - Radar SLAM trajectory vs. odometry comparison graph
  - 2D point cloud map of radar SLAM

Usage:
  python radar_slam.py --bag ~/datasets/03_80m_other_sensor.bag
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
from scipy.interpolate import interp1d


###############################################################################
# Math
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
# Transforms — identity rotation, translation only
###############################################################################

def get_radar_to_base():
    """Identity rotation, translation only."""
    T_r1 = np.eye(4)
    T_r1[:3, 3] = [0.565, 0.295, 0.3785]

    T_r2 = np.eye(4)
    T_r2[:3, 3] = [-0.565, -0.295, 0.3785]

    return T_r1, T_r2


###############################################################################
# Data extraction
###############################################################################

def extract_data(bag_path):
    print(f"[DATA] Reading {bag_path}...")
    bag = rosbag.Bag(bag_path)

    topics = ['/robot/robotnik_base_control/odom',
              '/isdr_driver_1/points', '/isdr_driver_2/points']

    odom_data, radar1_data, radar2_data = [], [], []

    for topic, msg, t in bag.read_messages(topics=topics):
        stamp = msg.header.stamp.to_sec()

        if topic == '/robot/robotnik_base_control/odom':
            pos = msg.pose.pose.position
            ori = msg.pose.pose.orientation
            q = [ori.x, ori.y, ori.z, ori.w]
            _, _, yaw = euler_from_quaternion(q)
            vx = msg.twist.twist.linear.x
            wz = msg.twist.twist.angular.z
            odom_data.append((stamp, pos.x, pos.y, pos.z, yaw, vx, wz))

        elif topic == '/isdr_driver_1/points':
            pts = np.array(list(pc2.read_points(
                msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)),
                dtype=np.float32)
            if len(pts) > 0:
                radar1_data.append((stamp, pts))

        elif topic == '/isdr_driver_2/points':
            pts = np.array(list(pc2.read_points(
                msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)),
                dtype=np.float32)
            if len(pts) > 0:
                radar2_data.append((stamp, pts))

    bag.close()
    odom_data.sort(key=lambda x: x[0])
    radar1_data.sort(key=lambda x: x[0])
    radar2_data.sort(key=lambda x: x[0])

    print(f"  Odom: {len(odom_data)} | Radar1: {len(radar1_data)} | Radar2: {len(radar2_data)}")
    return odom_data, radar1_data, radar2_data


###############################################################################
# Odom interpolator
###############################################################################

class OdomInterpolator:
    def __init__(self, odom_data):
        t = np.array([d[0] for d in odom_data])
        x = np.array([d[1] for d in odom_data])
        y = np.array([d[2] for d in odom_data])
        z = np.array([d[3] for d in odom_data])
        yaw = np.unwrap(np.array([d[4] for d in odom_data]))
        vx = np.array([d[5] for d in odom_data])

        self.f_x = interp1d(t, x, fill_value='extrapolate')
        self.f_y = interp1d(t, y, fill_value='extrapolate')
        self.f_z = interp1d(t, z, fill_value='extrapolate')
        self.f_yaw = interp1d(t, yaw, fill_value='extrapolate')
        self.f_vx = interp1d(t, vx, fill_value='extrapolate')
        self.x0, self.y0, self.z0 = x[0], y[0], z[0]
        self.yaw0 = yaw[0]

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
# Radar accumulation with motion compensation + outlier filtering
###############################################################################

def find_radar_hits(t_scan, radar_data, t_window):
    """Binary search for radar messages within +/- t_window."""
    if not radar_data:
        return []
    lo, hi = 0, len(radar_data) - 1
    t_lo, t_hi = t_scan - t_window, t_scan + t_window

    while lo < hi:
        mid = (lo + hi) // 2
        if radar_data[mid][0] < t_lo:
            lo = mid + 1
        else:
            hi = mid

    hits = []
    for i in range(lo, len(radar_data)):
        t_r = radar_data[i][0]
        if t_r > t_hi:
            break
        hits.append((t_r, radar_data[i][1]))
    return hits


def accumulate_compensated(radar_hits, t_scan, odom, T_radar_to_base,
                           min_range=0.5, max_range=6.0, min_intensity=-25.0):
    """
    Motion-compensate each radar point to base frame at t_scan.
    Per-point filtering:
      - Range gate: reject too-close (robot body) and too-far (multipath)
      - Intensity gate: reject weak returns (noise)
    """
    R_r2b = T_radar_to_base[:3, :3]
    t_r2b = T_radar_to_base[:3, 3]

    compensated = []
    for t_r, pts in radar_hits:
        xyz = pts[:, :3]
        intensity = pts[:, 3] if pts.shape[1] > 3 else None

        # Range filter in radar frame
        ranges = np.linalg.norm(xyz, axis=1)
        mask = (ranges >= min_range) & (ranges <= max_range)

        # Intensity filter
        if intensity is not None and min_intensity is not None:
            mask &= (intensity >= min_intensity)

        xyz = xyz[mask]
        if len(xyz) == 0:
            continue

        # Transform to base frame
        pts_base = (R_r2b @ xyz.T).T + t_r2b

        # Motion compensation: move from pose@t_r to pose@t_scan
        x1, y1, z1, yaw1 = odom.get_pose(t_r)
        x2, y2, z2, yaw2 = odom.get_pose(t_scan)
        T1 = pose_to_T(x1, y1, z1, yaw1)
        T2 = pose_to_T(x2, y2, z2, yaw2)
        T_rel = np.linalg.inv(T2) @ T1
        pts_comp = (T_rel[:3, :3] @ pts_base.T).T + T_rel[:3, 3]
        compensated.append(pts_comp)

    if compensated:
        return np.vstack(compensated)
    return None


def build_radar_scan(t_scan, radar1_data, radar2_data, odom,
                     T_r1, T_r2, t_window,
                     min_range=0.5, max_range=6.0, min_intensity=-20.0):
    """Build one filtered, motion-compensated radar scan."""
    all_pts = []
    for rdata, T_r in [(radar1_data, T_r1), (radar2_data, T_r2)]:
        hits = find_radar_hits(t_scan, rdata, t_window)
        if hits:
            pts = accumulate_compensated(hits, t_scan, odom, T_r,
                                          min_range, max_range, min_intensity)
            if pts is not None:
                all_pts.append(pts)
    if all_pts:
        return np.vstack(all_pts)
    return None


def filter_scan_statistical(pts, nb_neighbors=10, std_ratio=1.0):
    """Aggressive outlier removal: statistical + radius."""
    if pts is None or len(pts) < 20:
        return pts
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    # Statistical: remove points whose avg distance to neighbors is > std_ratio * std
    filtered, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio)

    # Radius: remove points with fewer than min_count neighbors within radius
    if len(filtered.points) > 20:
        filtered, _ = filtered.remove_radius_outlier(nb_points=5, radius=1.0)

    if len(filtered.points) < 5:
        return pts  # don't filter if too aggressive
    return np.asarray(filtered.points)


###############################################################################
# Radar submap
###############################################################################

class RadarSubmap:
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


###############################################################################
# Radar ICP — tuned for sparse 2D data
###############################################################################

def radar_match(src_pts, submap, T_init, max_dist=0.5):
    """
    Multi-resolution ICP for sparse radar:
      1. Coarse pass (max_dist) to pull in roughly
      2. Fine pass (max_dist/2) to refine
    Reject if fine fitness too low — means alignment isn't trustworthy.
    """
    target = submap.get()
    if target is None or len(target.points) < 30:
        return None, 0.0

    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(src_pts)
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

        # Use fine fitness as quality measure
        n_inliers = int(r2.fitness * len(src.points))
        if n_inliers < 10 or r2.fitness < 0.2:
            return None, r2.fitness

        return r2.transformation, r2.fitness
    except:
        return None, 0.0


###############################################################################
# Radar-only SLAM loop
###############################################################################

def run_radar_slam(odom_data, radar1_data, radar2_data,
                   t_window=4.0, scan_interval=0.5, submap_size=10,
                   min_range=0.5, max_range=5.0, min_intensity=-20.0):
    odom = OdomInterpolator(odom_data)
    T_r1, T_r2 = get_radar_to_base()

    all_times = [d[0] for d in radar1_data] + [d[0] for d in radar2_data]
    t_start = min(all_times) + t_window
    t_end = max(all_times) - t_window
    scan_times = np.arange(t_start, t_end, scan_interval)
    n_scans = len(scan_times)

    submap = RadarSubmap(max_scans=submap_size, voxel_size=0.15)

    x0, y0, z0, yaw0 = odom.get_pose(scan_times[0])
    global_pose = pose_to_T(x0, y0, z0, yaw0)

    trajectory = []
    map_pcds = []
    fitness_log = []
    pt_counts = []
    icp_used = 0
    odom_fallback = 0

    print(f"\n[RADAR SLAM] {n_scans} scans | interval={scan_interval}s | "
          f"window=+/-{t_window}s | submap={submap_size}")
    print(f"  Filters: range=[{min_range}, {max_range}]m, "
          f"intensity>={min_intensity} dB")

    t0 = time.time()
    for si in range(n_scans):
        t_scan = scan_times[si]

        # Build filtered radar scan
        radar_pts = build_radar_scan(
            t_scan, radar1_data, radar2_data, odom, T_r1, T_r2, t_window,
            min_range, max_range, min_intensity)

        # Statistical outlier removal
        if radar_pts is not None:
            radar_pts = filter_scan_statistical(radar_pts, nb_neighbors=8, std_ratio=1.5)

        n_pts = len(radar_pts) if radar_pts is not None else 0
        pt_counts.append(n_pts)

        if si == 0:
            trajectory.append((t_scan, global_pose.copy()))
            if radar_pts is not None:
                submap.add_scan(radar_pts, global_pose)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(radar_pts)
                map_pcds.append(copy.deepcopy(pcd).transform(global_pose))
            fitness_log.append(1.0)
            print(f"  [Scan 0] Init | {n_pts} pts | submap: {submap.size()}")
            continue

        # Odom prediction
        t_prev = scan_times[si - 1]
        T_odom_rel = odom.get_relative_transform(t_prev, t_scan)
        T_init = trajectory[-1][1] @ T_odom_rel

        # Try radar ICP
        T_result = T_init
        fitness = 0.0
        if radar_pts is not None and n_pts >= 15 and submap.size() >= 50:
            T_icp, fitness = radar_match(radar_pts, submap, T_init, max_dist=0.5)
            if T_icp is not None and fitness > 0.2:
                # Blend ICP + odom: adaptive weight based on fitness
                yaw_odom = rot_to_yaw(T_init[:3, :3])
                yaw_icp = rot_to_yaw(T_icp[:3, :3])

                w = min(0.5, fitness * 0.6)  # scale with fitness, cap at 0.5

                dyaw = np.arctan2(np.sin(yaw_icp - yaw_odom),
                                  np.cos(yaw_icp - yaw_odom))
                yaw_blend = yaw_odom + w * dyaw
                t_blend = (1 - w) * T_init[:3, 3] + w * T_icp[:3, 3]

                T_result = np.eye(4)
                T_result[:3, :3] = yaw_to_rot(yaw_blend)
                T_result[:3, 3] = t_blend
                icp_used += 1
            else:
                odom_fallback += 1
                fitness = 0.0
        else:
            odom_fallback += 1

        global_pose = T_result.copy()
        global_pose[2, 3] = 0.0
        global_pose[:3, :3] = yaw_to_rot(rot_to_yaw(global_pose[:3, :3]))

        trajectory.append((t_scan, global_pose.copy()))
        fitness_log.append(fitness)

        if radar_pts is not None:
            submap.add_scan(radar_pts, global_pose)

        if si % 5 == 0 and radar_pts is not None:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(radar_pts)
            map_pcds.append(copy.deepcopy(pcd).transform(global_pose))

        if si % 100 == 0:
            p = global_pose[:3, 3]
            src = "ICP" if fitness > 0.2 else "ODOM"
            print(f"  [Scan {si:4d}/{n_scans}] "
                  f"pos=({p[0]:7.2f}, {p[1]:7.2f}) "
                  f"{n_pts:3d} pts  fitness={fitness:.3f}  "
                  f"submap={submap.size():4d}  [{src}]")

    total_t = time.time() - t0
    print(f"\n[RADAR SLAM] Done. {n_scans} scans in {total_t:.1f}s")
    print(f"  ICP used: {icp_used}/{n_scans} ({100*icp_used/n_scans:.0f}%)")
    print(f"  Odom fallback: {odom_fallback}/{n_scans}")
    print(f"  Avg pts/scan: {np.mean(pt_counts):.1f} "
          f"(min={np.min(pt_counts)}, max={np.max(pt_counts)})")

    return trajectory, map_pcds, fitness_log, pt_counts


###############################################################################
# Save & visualize
###############################################################################

def save_results(trajectory, map_pcds, fitness_log, pt_counts, output_dir):
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
        print(f"[SAVE] Map: {len(combined.points)} pts")

    np.savetxt(os.path.join(output_dir, "fitness_log.txt"),
               np.array(fitness_log), fmt="%.4f")
    print(f"[SAVE] {len(traj_arr)} poses -> {output_dir}")
    return traj_arr


def visualize(traj_arr, odom_data, output_dir):
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

    # Trajectory comparison: Radar SLAM vs Odom
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.plot(slam_pos[:, 0], slam_pos[:, 1], 'b-', lw=2, label='Radar SLAM', alpha=0.8)
    ax.plot(odom_pos[:, 0], odom_pos[:, 1], 'r--', lw=1.5, alpha=0.6, label='Odometry')
    ax.plot(slam_pos[0, 0], slam_pos[0, 1], 'go', ms=12, label='Start', zorder=5)
    ax.plot(slam_pos[-1, 0], slam_pos[-1, 1], 'rs', ms=12, label='End', zorder=5)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('Radar SLAM Trajectory vs Odometry', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "trajectory_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"[VIS] Saved {path}")
    plt.close()

    # 2D map visualization
    map_path = os.path.join(output_dir, "global_map.pcd")
    if os.path.exists(map_path):
        pcd = o3d.io.read_point_cloud(map_path)
        pts = np.asarray(pcd.points)

        fig, ax = plt.subplots(figsize=(12, 10))
        ax.scatter(pts[:, 0], pts[:, 1], s=1, c='green', alpha=0.4, label='Radar map')
        ax.plot(slam_pos[:, 0], slam_pos[:, 1], 'b-', lw=2, alpha=0.7, label='SLAM trajectory')
        ax.plot(slam_pos[0, 0], slam_pos[0, 1], 'go', ms=12, label='Start', zorder=5)
        ax.plot(slam_pos[-1, 0], slam_pos[-1, 1], 'rs', ms=12, label='End', zorder=5)
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title('Radar SLAM 2D Map (Bird\'s Eye View)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(output_dir, "map_2d.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"[VIS] Saved {path}")
        plt.close()


###############################################################################
# Main
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="Radar-only SLAM")
    parser.add_argument("--bag", required=True)
    parser.add_argument("--window", type=float, default=4.0)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--submap-size", type=int, default=10)
    parser.add_argument("--output", default="radar_slam_output")
    parser.add_argument("--min-range", type=float, default=0.5)
    parser.add_argument("--max-range", type=float, default=5.0)
    parser.add_argument("--min-intensity", type=float, default=-20.0)
    args = parser.parse_args()

    odom_data, radar1_data, radar2_data = extract_data(args.bag)

    trajectory, map_pcds, fitness_log, pt_counts = run_radar_slam(
        odom_data, radar1_data, radar2_data,
        t_window=args.window, scan_interval=args.interval,
        submap_size=args.submap_size,
        min_range=args.min_range, max_range=args.max_range,
        min_intensity=args.min_intensity)

    traj_arr = save_results(trajectory, map_pcds, fitness_log, pt_counts, args.output)
    visualize(traj_arr, odom_data, args.output)

    d = np.linalg.norm(traj_arr[-1, 1:4] - traj_arr[0, 1:4])
    print(f"\n[RESULT] Return-to-origin error: {d:.2f}m")
    print(f"[RESULT] Results saved to: {args.output}")


if __name__ == "__main__":
    main()
