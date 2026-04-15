#!/usr/bin/env python3
"""
View SLAM Maps
================
View one or more SLAM maps with optional GT overlay.
GT is automatically aligned to the first map.

Usage:
  python view_map.py slam_output/baseline
  python view_map.py slam_output/baseline slam_output/standard_degraded --gt
  python view_map.py slam_output/standard_degraded slam_output/standard_degraded_radar --gt
"""

import argparse
import numpy as np
import open3d as o3d
import os

GT_CACHE_PATH = "slam_output/gt_cache.pcd"

MODE_COLORS = {
    'baseline':                ([0.12, 0.47, 0.71], "Blue"),
    'standard_degraded':       ([1.00, 0.50, 0.05], "Orange"),
    'heavy_degraded':          ([0.84, 0.15, 0.16], "Red"),
    'radar':                   ([0.17, 0.63, 0.17], "Green"),
    'standard_degraded_radar': ([0.09, 0.75, 0.81], "Cyan"),
    'heavy_degraded_radar':    ([0.58, 0.40, 0.74], "Purple"),
    'odom_only':               ([0.90, 0.90, 0.10], "Yellow"),
}

FALLBACK_PALETTE = [
    ([0.12, 0.47, 0.71], "Blue"),
    ([0.84, 0.15, 0.16], "Red"),
    ([0.17, 0.63, 0.17], "Green"),
    ([0.58, 0.40, 0.74], "Purple"),
    ([1.00, 0.50, 0.05], "Orange"),
    ([0.09, 0.75, 0.81], "Cyan"),
    ([0.90, 0.90, 0.10], "Yellow"),
    ([0.90, 0.40, 0.60], "Pink"),
]


def align_gt_to_slam(gt_pcd, slam_pcd, voxel_size=0.5):
    """RANSAC + ICP to bring GT into SLAM coordinate frame.
    Runs RANSAC 5 times with 500k iterations, keeps best."""
    print("[ALIGN] Aligning GT to SLAM frame (5 attempts)...")
    gt_d = gt_pcd.voxel_down_sample(voxel_size)
    slam_d = slam_pcd.voxel_down_sample(voxel_size)

    for p in [gt_d, slam_d]:
        p.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 3, max_nn=30))

    gt_f = o3d.pipelines.registration.compute_fpfh_feature(
        gt_d, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
    slam_f = o3d.pipelines.registration.compute_fpfh_feature(
        slam_d, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))

    best_T = np.eye(4)
    best_fitness = -1

    for attempt in range(5):
        ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            gt_d, slam_d, gt_f, slam_f, True, voxel_size * 2,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(), 3,
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
             o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 2)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(500000, 0.9999))

        icp = o3d.pipelines.registration.registration_icp(
            gt_d, slam_d, voxel_size * 1.5, ransac.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))

        if icp.fitness > best_fitness:
            best_fitness = icp.fitness
            best_T = icp.transformation

    print(f"  Best ICP fitness: {best_fitness:.4f}")
    gt_aligned = o3d.geometry.PointCloud(gt_pcd)
    gt_aligned.transform(best_T)
    return gt_aligned


def main():
    parser = argparse.ArgumentParser(description="View SLAM maps")
    parser.add_argument("dirs", nargs='+',
                        help="SLAM output directories (e.g. slam_output/baseline)")
    parser.add_argument("--gt", action="store_true",
                        help="Overlay ground truth")
    args = parser.parse_args()

    geometries = []
    legend = []
    first_pcd = None

    fallback_idx = 0
    for d in args.dirs:
        label = os.path.basename(d.rstrip('/'))
        if label in MODE_COLORS:
            color, color_name = MODE_COLORS[label]
        else:
            color, color_name = FALLBACK_PALETTE[fallback_idx % len(FALLBACK_PALETTE)]
            fallback_idx += 1

        # Load map
        map_path = os.path.join(d, "global_map.pcd")
        if os.path.exists(map_path):
            pcd = o3d.io.read_point_cloud(map_path)
            pcd.paint_uniform_color(color)
            geometries.append(pcd)
            legend.append((label, color_name))
            print(f"[MAP] {color_name:<8} {label}: {len(pcd.points)} pts")
            if first_pcd is None:
                first_pcd = o3d.geometry.PointCloud(pcd)
        else:
            print(f"[MAP] {label}: no global_map.pcd")

        # Load trajectory
        traj_path = os.path.join(d, "trajectory.txt")
        if os.path.exists(traj_path):
            data = np.loadtxt(traj_path)
            positions = data[:, 1:4]
            lines = [[j, j + 1] for j in range(len(positions) - 1)]
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(positions)
            ls.lines = o3d.utility.Vector2iVector(lines)
            bright = [min(1.0, c + 0.3) for c in color]
            ls.colors = o3d.utility.Vector3dVector([bright] * len(lines))
            geometries.append(ls)

    # GT
    if args.gt and first_pcd is not None:
        if not os.path.exists(GT_CACHE_PATH):
            print(f"[GT] No cache at {GT_CACHE_PATH}")
            print(f"  Run: python compare.py --gt /path/to/GT.las")
        else:
            gt_raw = o3d.io.read_point_cloud(GT_CACHE_PATH)
            gt_aligned = align_gt_to_slam(gt_raw, first_pcd)
            gt_aligned.paint_uniform_color([0.5, 0.5, 0.5])
            geometries.append(gt_aligned)
            legend.append(("Ground Truth", "Gray"))
            print(f"[GT]  Gray     Ground Truth: {len(gt_aligned.points)} pts")

    if not geometries:
        print("Nothing to display.")
        return

    # Legend
    print(f"\n{'='*35}")
    for label, color_name in legend:
        print(f"  {color_name:<10} {label}")
    print(f"{'='*35}")

    # Viewer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Map Viewer", width=1400, height=900)
    for g in geometries:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0.1, 0.1, 0.1])
    opt.show_coordinate_frame = True
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
