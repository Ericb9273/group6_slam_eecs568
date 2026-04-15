#!/usr/bin/env python3
"""
Radar SLAM Parameter Sweep
============================
Runs radar-only SLAM with different parameter combos and plots
all trajectories for visual comparison.

Usage:
  python radar_param_sweep.py --bag ~/datasets/03_80m_other_sensor.bag
"""

import argparse
import numpy as np
from collections import deque
from itertools import product
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


def get_radar_to_base():
    T_r1 = np.eye(4); T_r1[:3, 3] = [0.565, 0.295, 0.3785]
    T_r2 = np.eye(4); T_r2[:3, 3] = [-0.565, -0.295, 0.3785]
    return T_r1, T_r2


###############################################################################
# Data extraction
###############################################################################

def extract_data(bag_path):
    print(f"[DATA] Reading {bag_path}...")
    bag = rosbag.Bag(bag_path)
    topics = ['/robot/robotnik_base_control/odom',
              '/isdr_driver_1/points', '/isdr_driver_2/points']
    odom_data, r1_data, r2_data = [], [], []

    for topic, msg, t in bag.read_messages(topics=topics):
        stamp = msg.header.stamp.to_sec()
        if topic == '/robot/robotnik_base_control/odom':
            pos = msg.pose.pose.position
            ori = msg.pose.pose.orientation
            _, _, yaw = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
            vx = msg.twist.twist.linear.x
            odom_data.append((stamp, pos.x, pos.y, pos.z, yaw, vx, 0))
        elif topic == '/isdr_driver_1/points':
            pts = np.array(list(pc2.read_points(
                msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)),
                dtype=np.float32)
            if len(pts) > 0:
                r1_data.append((stamp, pts))
        elif topic == '/isdr_driver_2/points':
            pts = np.array(list(pc2.read_points(
                msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)),
                dtype=np.float32)
            if len(pts) > 0:
                r2_data.append((stamp, pts))

    bag.close()
    for d in [odom_data, r1_data, r2_data]:
        d.sort(key=lambda x: x[0])
    print(f"  Odom: {len(odom_data)} | R1: {len(r1_data)} | R2: {len(r2_data)}")
    return odom_data, r1_data, r2_data


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
        self.f_x = interp1d(t, x, fill_value='extrapolate')
        self.f_y = interp1d(t, y, fill_value='extrapolate')
        self.f_z = interp1d(t, z, fill_value='extrapolate')
        self.f_yaw = interp1d(t, yaw, fill_value='extrapolate')
        self.x0, self.y0, self.z0 = x[0], y[0], z[0]
        self.yaw0 = yaw[0]

    def get_pose(self, t):
        dx = float(self.f_x(t)) - self.x0
        dy = float(self.f_y(t)) - self.y0
        dz = float(self.f_z(t)) - self.z0
        yaw = float(self.f_yaw(t)) - self.yaw0
        c, s = np.cos(-self.yaw0), np.sin(-self.yaw0)
        return c*dx - s*dy, s*dx + c*dy, dz, yaw

    def get_relative_transform(self, t1, t2):
        x1, y1, z1, yaw1 = self.get_pose(t1)
        x2, y2, z2, yaw2 = self.get_pose(t2)
        return np.linalg.inv(pose_to_T(x1, y1, z1, yaw1)) @ pose_to_T(x2, y2, z2, yaw2)


###############################################################################
# Radar helpers
###############################################################################

def find_radar_hits(t_scan, radar_data, t_window):
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
        if radar_data[i][0] > t_hi:
            break
        hits.append((radar_data[i][0], radar_data[i][1]))
    return hits


def accumulate_compensated(hits, t_scan, odom, T_r2b,
                           min_range=0.5, max_range=5.0, min_intensity=-20.0):
    R, t = T_r2b[:3, :3], T_r2b[:3, 3]
    compensated = []
    for t_r, pts in hits:
        xyz = pts[:, :3]
        intensity = pts[:, 3] if pts.shape[1] > 3 else None
        ranges = np.linalg.norm(xyz, axis=1)
        mask = (ranges >= min_range) & (ranges <= max_range)
        if intensity is not None:
            mask &= (intensity >= min_intensity)
        xyz = xyz[mask]
        if len(xyz) == 0:
            continue
        pts_base = (R @ xyz.T).T + t
        x1, y1, z1, yaw1 = odom.get_pose(t_r)
        x2, y2, z2, yaw2 = odom.get_pose(t_scan)
        T_rel = np.linalg.inv(pose_to_T(x2, y2, z2, yaw2)) @ pose_to_T(x1, y1, z1, yaw1)
        compensated.append((T_rel[:3, :3] @ pts_base.T).T + T_rel[:3, 3])
    return np.vstack(compensated) if compensated else None


def build_radar_scan(t_scan, r1, r2, odom, T_r1, T_r2, p):
    all_pts = []
    for rdata, T_r in [(r1, T_r1), (r2, T_r2)]:
        hits = find_radar_hits(t_scan, rdata, p['t_window'])
        if hits:
            pts = accumulate_compensated(hits, t_scan, odom, T_r,
                                          p['min_range'], p['max_range'],
                                          p['min_intensity'])
            if pts is not None:
                all_pts.append(pts)
    return np.vstack(all_pts) if all_pts else None


def filter_outliers(pts, nb_neighbors=10, std_ratio=1.0):
    if pts is None or len(pts) < 20:
        return pts
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    if len(pcd.points) > 20:
        pcd, _ = pcd.remove_radius_outlier(nb_points=5, radius=1.0)
    return np.asarray(pcd.points) if len(pcd.points) > 5 else None


###############################################################################
# Submap
###############################################################################

class RadarSubmap:
    def __init__(self, max_scans, voxel_size):
        self.max_scans = max_scans
        self.voxel_size = voxel_size
        self.scan_queue = deque()
        self.combined = None

    def add_scan(self, pts, T_world):
        if pts is None or len(pts) < 3:
            return
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        self.scan_queue.append(pcd.transform(T_world))
        while len(self.scan_queue) > self.max_scans:
            self.scan_queue.popleft()
        combined = o3d.geometry.PointCloud()
        for p in self.scan_queue:
            combined += p
        if len(combined.points) > 10:
            combined = combined.voxel_down_sample(self.voxel_size)
        self.combined = combined

    def get(self):
        return self.combined

    def size(self):
        return len(self.combined.points) if self.combined else 0


###############################################################################
# Single SLAM run
###############################################################################

def run_one(odom, r1_data, r2_data, T_r1, T_r2, p):
    """Run radar SLAM with parameter dict p. Returns trajectory as Nx2 array."""
    all_times = [d[0] for d in r1_data] + [d[0] for d in r2_data]
    t_start = min(all_times) + p['t_window']
    t_end = max(all_times) - p['t_window']
    scan_times = np.arange(t_start, t_end, 0.5)
    n = len(scan_times)

    submap = RadarSubmap(p['submap_size'], p['submap_voxel'])
    x0, y0, z0, yaw0 = odom.get_pose(scan_times[0])
    gp = pose_to_T(x0, y0, z0, yaw0)

    traj = []
    icp_used = 0

    for si in range(n):
        t_scan = scan_times[si]
        pts = build_radar_scan(t_scan, r1_data, r2_data, odom, T_r1, T_r2, p)
        pts = filter_outliers(pts)
        n_pts = len(pts) if pts is not None else 0

        if si == 0:
            traj.append([gp[0, 3], gp[1, 3]])
            if pts is not None:
                submap.add_scan(pts, gp)
            continue

        T_odom_rel = odom.get_relative_transform(scan_times[si-1], t_scan)
        T_init = gp @ T_odom_rel
        T_result = T_init

        if pts is not None and n_pts >= 15 and submap.size() >= p['min_submap_pts']:
            src = o3d.geometry.PointCloud()
            src.points = o3d.utility.Vector3dVector(pts)
            target = submap.get()
            if target is not None and len(target.points) >= 30:
                try:
                    # Coarse
                    r1 = o3d.pipelines.registration.registration_icp(
                        src, target, p['max_dist'], T_init,
                        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                        o3d.pipelines.registration.ICPConvergenceCriteria(
                            max_iteration=30))
                    # Fine
                    r2 = o3d.pipelines.registration.registration_icp(
                        src, target, p['max_dist'] * 0.5, r1.transformation,
                        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                        o3d.pipelines.registration.ICPConvergenceCriteria(
                            max_iteration=30))

                    n_inliers = int(r2.fitness * len(src.points))
                    if n_inliers >= 10 and r2.fitness > p['min_fitness']:
                        w = min(p['max_weight'], r2.fitness * 0.6)
                        yaw_o = rot_to_yaw(T_init[:3, :3])
                        yaw_i = rot_to_yaw(r2.transformation[:3, :3])
                        dyaw = np.arctan2(np.sin(yaw_i - yaw_o),
                                          np.cos(yaw_i - yaw_o))
                        T_result = np.eye(4)
                        T_result[:3, :3] = yaw_to_rot(yaw_o + w * dyaw)
                        T_result[:3, 3] = (1-w) * T_init[:3, 3] + w * r2.transformation[:3, 3]
                        icp_used += 1
                except:
                    pass

        gp = T_result.copy()
        gp[2, 3] = 0.0
        gp[:3, :3] = yaw_to_rot(rot_to_yaw(gp[:3, :3]))
        traj.append([gp[0, 3], gp[1, 3]])

        if pts is not None:
            submap.add_scan(pts, gp)

    return np.array(traj), icp_used, n


###############################################################################
# Parameter grid
###############################################################################

# Each param: (name_for_display, dict_key, [values_to_try])
PARAM_GRID = {
    't_window':      [2.0, 4.0, 6.0],
    'max_dist':      [0.3, 0.5, 1.0],
    'submap_size':   [5, 10, 20],
    'submap_voxel':  [0.15],
    'min_range':     [0.5],
    'max_range':     [5.0],
    'min_intensity': [-20.0],
    'min_submap_pts': [40],
    'min_fitness':   [0.2],
    'max_weight':    [0.5],
}

# We sweep over the 3 most impactful params, fix the rest
SWEEP_KEYS = ['t_window', 'max_dist', 'submap_size']


def build_param_combos():
    """Build all combinations of sweep params."""
    sweep_vals = [PARAM_GRID[k] for k in SWEEP_KEYS]
    combos = []
    for vals in product(*sweep_vals):
        p = {}
        for k, v in PARAM_GRID.items():
            if k in SWEEP_KEYS:
                p[k] = vals[SWEEP_KEYS.index(k)]
            else:
                p[k] = v[0]  # use first (only) value
        combos.append(p)
    return combos


###############################################################################
# Main
###############################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True)
    parser.add_argument("--output", default="slam_output/param_sweep")
    args = parser.parse_args()

    odom_data, r1_data, r2_data = extract_data(args.bag)
    odom = OdomInterpolator(odom_data)
    T_r1, T_r2 = get_radar_to_base()

    # Odom trajectory for reference
    odom_traj = []
    for d in odom_data[::20]:
        x, y, _, _ = odom.get_pose(d[0])
        odom_traj.append([x, y])
    odom_traj = np.array(odom_traj)

    combos = build_param_combos()
    print(f"\n[SWEEP] {len(combos)} parameter combinations")
    print(f"  t_window:    {PARAM_GRID['t_window']}")
    print(f"  max_dist:    {PARAM_GRID['max_dist']}")
    print(f"  submap_size: {PARAM_GRID['submap_size']}")

    results = []
    for i, p in enumerate(combos):
        label = f"tw={p['t_window']:.0f} md={p['max_dist']:.1f} sm={p['submap_size']}"
        print(f"\n[{i+1}/{len(combos)}] {label}")
        t0 = time.time()
        traj, icp_used, n_scans = run_one(odom, r1_data, r2_data, T_r1, T_r2, p)
        dt = time.time() - t0
        print(f"  ICP: {icp_used}/{n_scans} ({100*icp_used/n_scans:.0f}%) | {dt:.1f}s")
        results.append((p, label, traj, icp_used, n_scans))

    # Plot grid: rows=t_window, cols=submap_size, color=max_dist
    tw_vals = PARAM_GRID['t_window']
    sm_vals = PARAM_GRID['submap_size']
    md_vals = PARAM_GRID['max_dist']
    md_colors = {0.3: 'blue', 0.5: 'green', 1.0: 'red'}

    fig, axes = plt.subplots(len(tw_vals), len(sm_vals),
                              figsize=(8 * len(sm_vals), 8 * len(tw_vals)))
    if len(tw_vals) == 1:
        axes = axes[np.newaxis, :]
    if len(sm_vals) == 1:
        axes = axes[:, np.newaxis]

    for p, label, traj, icp_used, n_scans in results:
        row = tw_vals.index(p['t_window'])
        col = sm_vals.index(p['submap_size'])
        ax = axes[row, col]
        color = md_colors.get(p['max_dist'], 'gray')
        ax.plot(traj[:, 0], traj[:, 1], color=color, lw=1.5, alpha=0.8,
                label=f"md={p['max_dist']:.1f} icp={100*icp_used/n_scans:.0f}%")

    for row, tw in enumerate(tw_vals):
        for col, sm in enumerate(sm_vals):
            ax = axes[row, col]
            ax.plot(odom_traj[:, 0], odom_traj[:, 1], 'r--', lw=1, alpha=0.4, label='Odom')
            ax.plot(odom_traj[0, 0], odom_traj[0, 1], 'ko', ms=8)
            ax.set_title(f'window=±{tw:.0f}s  submap={sm}', fontsize=12, fontweight='bold')
            ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc='upper left')
            if col == 0:
                ax.set_ylabel('Y (m)')
            if row == len(tw_vals) - 1:
                ax.set_xlabel('X (m)')

    plt.suptitle('Radar SLAM Parameter Sweep\n'
                 'Blue=max_dist 0.3  Green=0.5  Red=1.0',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    os.makedirs(args.output, exist_ok=True)
    path = os.path.join(args.output, "param_sweep.png")
    plt.savefig(path, dpi=120, bbox_inches='tight')
    print(f"\n[VIS] Saved {path}")

    # Summary table
    print(f"\n{'='*75}")
    print(f"{'Config':<35} {'ICP%':>6} {'Scans':>6}")
    print(f"{'-'*75}")
    for p, label, traj, icp_used, n_scans in results:
        print(f"{label:<35} {100*icp_used/n_scans:>5.0f}% {n_scans:>6}")
    print(f"{'='*75}")

    plt.show()


if __name__ == "__main__":
    main()
