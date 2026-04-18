#!/usr/bin/env python3
"""
Compare All SLAM Results
==========================
Auto-detects all modes in slam_output/, generates all comparison plots.
Skips radar_only for GT comparison (2D vs 3D mismatch).

Usage:
  python compare.py                          # uses slam_output/
  python compare.py --dir slam_output
  python compare.py --gt ~/datasets/GT.las   # first time only
"""

import argparse
import numpy as np
import open3d as o3d
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

try:
    import laspy
    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False

GT_CACHE_PATH = "slam_output/gt_cache.pcd"

COLORS = {
    'baseline':              ('tab:blue',   'Baseline (LiDAR)'),
    'standard_degraded':     ('tab:orange', 'Standard Degraded'),
    'heavy_degraded':        ('tab:red',    'Heavy Degraded'),
    'radar':                 ('tab:green',  'LiDAR + Radar'),
    'standard_degraded_radar': ('tab:cyan',   'Standard Degraded + Radar'),
    'heavy_degraded_radar':  ('tab:purple', 'Heavy Degraded + Radar'),
    'radar_only':            ('goldenrod',  'Radar Only'),
    'odom_only':             ('tab:gray',   'Odom Only'),
}

# Modes to skip for GT comparison (2D radar can't compare to 3D GT)
SKIP_GT = set()


###############################################################################
# Helpers
###############################################################################

def find_modes(slam_dir):
    modes = []
    for name in sorted(os.listdir(slam_dir)):
        traj = os.path.join(slam_dir, name, "trajectory.txt")
        if os.path.exists(traj):
            modes.append(name)
    return modes


def load_trajectory(traj_path):
    data = np.loadtxt(traj_path)
    return data[:, 0], data[:, 1:4]


def load_fitness(fit_path):
    if os.path.exists(fit_path):
        return np.loadtxt(fit_path)
    return None


###############################################################################
# GT
###############################################################################

def load_or_cache_gt(las_path=None, cache_path=GT_CACHE_PATH, subsample=10):
    if os.path.exists(cache_path):
        print(f"[GT] Loading from cache: {cache_path}")
        pcd = o3d.io.read_point_cloud(cache_path)
        print(f"  {len(pcd.points)} points")
        return pcd

    if las_path is None:
        return None

    if not HAS_LASPY:
        print("Install laspy: pip install laspy")
        return None

    print(f"[GT] Loading LAS: {las_path}...")
    points_list = []
    total = 0
    with laspy.open(las_path) as reader:
        for chunk in reader.chunk_iterator(1_000_000):
            chunk_pts = np.vstack([chunk.x, chunk.y, chunk.z]).T
            points_list.append(chunk_pts[::subsample])
            total += len(chunk_pts)

    points = np.vstack(points_list)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd = pcd.voxel_down_sample(0.3)
    print(f"  {len(pcd.points)} points after downsample")

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    o3d.io.write_point_cloud(cache_path, pcd)
    return pcd


def center_cloud(pcd):
    pts = np.asarray(pcd.points)
    centroid = pts.mean(axis=0)
    pcd_c = o3d.geometry.PointCloud()
    pcd_c.points = o3d.utility.Vector3dVector(pts - centroid)
    return pcd_c, centroid


def flatten_to_2d(pcd):
    """Project point cloud to XY plane (set Z=0)."""
    pts = np.asarray(pcd.points).copy()
    pts[:, 2] = 0.0
    pcd_2d = o3d.geometry.PointCloud()
    pcd_2d.points = o3d.utility.Vector3dVector(pts)
    return pcd_2d


def align_slam_to_gt(gt_pcd, slam_pcd, voxel=0.5):
    """Align a SLAM map to GT using RANSAC+ICP in 3D. High iterations for consistency.
    Returns (gt_centered, slam_centered, transform)."""
    gt_c, gt_cent = center_cloud(gt_pcd)
    slam_c, slam_cent = center_cloud(slam_pcd)

    gt_d = gt_c.voxel_down_sample(voxel)
    slam_d = slam_c.voxel_down_sample(voxel)

    for p in [gt_d, slam_d]:
        p.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel * 3, max_nn=30))

    gt_f = o3d.pipelines.registration.compute_fpfh_feature(
        gt_d, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100))
    slam_f = o3d.pipelines.registration.compute_fpfh_feature(
        slam_d, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100))

    # Run RANSAC multiple times, keep best
    best_T = np.eye(4)
    best_fitness = -1
    for attempt in range(5):
        ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            slam_d, gt_d, slam_f, gt_f, True, voxel * 2,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(), 3,
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
             o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel * 2)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(500000, 0.9999))

        # Refine each RANSAC result with ICP
        icp = o3d.pipelines.registration.registration_icp(
            slam_d, gt_d, voxel * 1.5, ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))

        if icp.fitness > best_fitness:
            best_fitness = icp.fitness
            best_T = icp.transformation

    print(f"    Best ICP fitness: {best_fitness:.4f} (5 RANSAC attempts)")
    return gt_c, slam_c, best_T, gt_cent, slam_cent


def compute_distances(gt_c, slam_aligned, max_dist=5.0):
    """Compute nearest-neighbor distances from SLAM points to GT."""
    gt_tree = o3d.geometry.KDTreeFlann(gt_c)
    slam_pts = np.asarray(slam_aligned.points)
    distances = []
    for i in range(len(slam_pts)):
        [_, _, d2] = gt_tree.search_knn_vector_3d(slam_pts[i], 1)
        d = np.sqrt(d2[0])
        if d < max_dist:
            distances.append(d)
    return np.array(distances)


def compare_all_gt(gt_pcd, slam_dir, gt_modes, voxel=0.5):
    """Align using baseline (or first available), then evaluate all modes
    in the same coordinate frame for fair comparison.
    
    Key: all modes are centered using the SAME centroid (from alignment mode),
    so the transform is valid for all of them."""

    # Find the best map to align with (prefer baseline)
    align_mode = None
    for pref in ['baseline', 'radar']:
        if pref in gt_modes:
            align_mode = pref
            break
    if align_mode is None:
        align_mode = gt_modes[0]

    align_path = os.path.join(slam_dir, align_mode, "global_map.pcd")
    align_pcd = o3d.io.read_point_cloud(align_path)
    print(f"\n[GT] Aligning using '{align_mode}' as reference ({len(align_pcd.points)} pts)")

    gt_c, align_c, T_align, gt_cent, slam_cent = align_slam_to_gt(gt_pcd, align_pcd, voxel)

    results = {}
    for mode in gt_modes:
        map_path = os.path.join(slam_dir, mode, "global_map.pcd")
        if not os.path.exists(map_path):
            continue

        slam_pcd = o3d.io.read_point_cloud(map_path)
        print(f"  {mode}: {len(slam_pcd.points)} pts")

        # Center using the SAME centroid as alignment mode
        pts = np.asarray(slam_pcd.points)
        slam_c_same = o3d.geometry.PointCloud()
        slam_c_same.points = o3d.utility.Vector3dVector(pts - slam_cent)
        slam_aligned = slam_c_same.transform(T_align)

        distances = compute_distances(gt_c, slam_aligned)
        if len(distances) > 0:
            results[mode] = {
                'mean': distances.mean(),
                'median': np.median(distances),
                'rmse': np.sqrt(np.mean(distances ** 2)),
                'p90': np.percentile(distances, 90),
                'p95': np.percentile(distances, 95),
            }

    return results


###############################################################################
# Plots
###############################################################################

def plot_trajectories_2d(slam_dir, modes):
    fig, ax = plt.subplots(figsize=(10, 10))
    for mode in modes:
        t, pos = load_trajectory(os.path.join(slam_dir, mode, "trajectory.txt"))
        color, label = COLORS.get(mode, ('gray', mode))
        ax.plot(pos[:, 0], pos[:, 1], color=color, lw=2, label=label)
    ax.plot(0, 0, 'ko', ms=10, zorder=5, label='Start')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_title('Trajectory Comparison (Bird\'s Eye)')
    ax.legend(fontsize=11); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(slam_dir, "comparison_2d.png"), dpi=150)
    plt.close()


def plot_trajectories_3d(slam_dir, modes):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    for mode in modes:
        t, pos = load_trajectory(os.path.join(slam_dir, mode, "trajectory.txt"))
        color, label = COLORS.get(mode, ('gray', mode))
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=color, lw=2, label=label)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('3D Trajectory Comparison'); ax.legend()
    plt.savefig(os.path.join(slam_dir, "comparison_3d.png"), dpi=150)
    plt.close()


def plot_height(slam_dir, modes):
    fig, ax = plt.subplots(figsize=(14, 5))
    for mode in modes:
        t, pos = load_trajectory(os.path.join(slam_dir, mode, "trajectory.txt"))
        color, label = COLORS.get(mode, ('gray', mode))
        ax.plot(t - t[0], pos[:, 2], color=color, lw=1.5, label=label)
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Z (m)')
    ax.set_title('Height Profile'); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(slam_dir, "comparison_height.png"), dpi=150)
    plt.close()


def plot_fitness(slam_dir, modes):
    fig, ax = plt.subplots(figsize=(14, 5))
    for mode in modes:
        fitness = load_fitness(os.path.join(slam_dir, mode, "fitness_log.txt"))
        if fitness is None:
            continue
        color, label = COLORS.get(mode, ('gray', mode))
        window = min(50, len(fitness) // 10 + 1)
        if window > 1:
            smooth = np.convolve(fitness, np.ones(window) / window, mode='valid')
            ax.plot(smooth, color=color, lw=1.5, label=f'{label} (avg)')
        else:
            ax.plot(fitness, color=color, lw=1, alpha=0.7, label=label)
    ax.set_xlabel('Scan Index'); ax.set_ylabel('ICP Fitness')
    ax.set_title('ICP Fitness Over Time'); ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(slam_dir, "comparison_fitness.png"), dpi=150)
    plt.close()


def plot_gt_comparison(slam_dir, gt_results):
    if not gt_results:
        return
    modes = list(gt_results.keys())
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(modes))
    width = 0.2
    bars1 = ax.bar(x - 1.5 * width, [gt_results[m]['mean'] for m in modes], width, label='Mean', color='steelblue')
    bars2 = ax.bar(x - 0.5 * width, [gt_results[m]['median'] for m in modes], width, label='Median', color='orange')
    bars3 = ax.bar(x + 0.5 * width, [gt_results[m]['rmse'] for m in modes], width, label='RMSE', color='green')
    bars4 = ax.bar(x + 1.5 * width, [gt_results[m]['p90'] for m in modes], width, label='90th percentile threshold', color='red')

    # Add value labels on bars
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    labels = [COLORS.get(m, ('', m))[1] for m in modes]
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel('Distance to GT (m)')
    ax.set_title('SLAM Map Accuracy vs Ground Truth')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(slam_dir, "accuracy_comparison.png"), dpi=150)
    plt.close()


def print_summary(slam_dir, modes):
    print(f"\n{'='*75}")
    print(f"{'Mode':<25} {'Scans':>6} {'Avg Fitness':>12}")
    print(f"{'-'*75}")
    for mode in modes:
        t, pos = load_trajectory(os.path.join(slam_dir, mode, "trajectory.txt"))
        fitness = load_fitness(os.path.join(slam_dir, mode, "fitness_log.txt"))
        avg_f = np.mean(fitness) if fitness is not None else 0
        _, label = COLORS.get(mode, ('', mode))
        print(f"{label:<25} {len(t):>6} {avg_f:>12.3f}")
    print(f"{'='*75}")


###############################################################################
# Main
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="Compare all SLAM results")
    parser.add_argument("--dir", default="slam_output")
    parser.add_argument("--gt", default=None,
                        help="Path to GT LAS file (first time only)")
    parser.add_argument("--cache", default=GT_CACHE_PATH)
    args = parser.parse_args()

    modes = find_modes(args.dir)
    if not modes:
        print(f"No results in {args.dir}/")
        return

    print(f"[COMPARE] Found: {modes}")
    print_summary(args.dir, modes)

    # Trajectory plots
    print("\n[PLOTS] Generating trajectory comparisons...")
    plot_trajectories_2d(args.dir, modes)
    plot_trajectories_3d(args.dir, modes)

    # GT comparison (3D, align once with baseline, skip radar_only/odom_only)
    gt_pcd = load_or_cache_gt(args.gt, args.cache)
    if gt_pcd is not None:
        gt_modes = [m for m in modes if m not in SKIP_GT]
        if gt_modes:
            print(f"\n[GT] Comparing against ground truth (3D)...")
            print(f"  Modes: {gt_modes}  (skipping: {[m for m in modes if m in SKIP_GT]})")
            gt_results = compare_all_gt(gt_pcd, args.dir, gt_modes)

            if gt_results:
                print(f"\n{'='*65}")
                print(f"{'Mode':<20} {'Mean':>8} {'Median':>8} {'RMSE':>8} {'90th':>8} {'95th':>8}")
                print(f"{'-'*65}")
                for mode in gt_results:
                    r = gt_results[mode]
                    print(f"{mode:<20} {r['mean']:>8.4f} {r['median']:>8.4f} "
                          f"{r['rmse']:>8.4f} {r['p90']:>8.4f} {r['p95']:>8.4f}")
                print(f"{'='*65}")
                plot_gt_comparison(args.dir, gt_results)
    else:
        print("\n[GT] No GT cache found. Run with --gt /path/to/GT.las first time.")

    print(f"\n[DONE] All plots saved to {args.dir}/")


if __name__ == "__main__":
    main()
