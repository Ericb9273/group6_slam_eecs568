#!/usr/bin/env python3
"""
Compare SLAM Map vs Ground Truth
==================================
Loads ground truth from a LAS file and the SLAM map from a PCD file,
aligns them using ICP, and computes cloud-to-cloud distance metrics.

The ground truth is likely in UTM/survey coordinates (large numbers),
so we center both clouds before alignment.

Usage:
  python compare_gt.py --gt 3D_point_cloud_GT.las --slam slam_output/global_map.pcd
  python compare_gt.py --gt 3D_point_cloud_GT.las --slam slam_output/global_map.pcd --view

Dependencies:
  pip install laspy open3d numpy matplotlib
"""

import argparse
import numpy as np
import open3d as o3d
import os

try:
    import laspy
except ImportError:
    print("Install laspy: pip install laspy")
    exit(1)

import matplotlib.pyplot as plt


###############################################################################
# 1. Load LAS file
###############################################################################

def load_las(path, subsample=10):
    """
    Load a LAS/LAZ file using chunked reading to avoid OOM.
    The GT file is 4.2 GB — too large to load at once.
    We read in chunks and keep every Nth point.
    """
    print(f"[GT] Loading {path} (every {subsample}th point, chunked)...")

    points_list = []
    total = 0

    with laspy.open(path) as reader:
        print(f"  Point format: {reader.header.point_format}")
        print(f"  Total points: {reader.header.point_count}")

        for chunk in reader.chunk_iterator(1_000_000):
            chunk_pts = np.vstack([chunk.x, chunk.y, chunk.z]).T
            # Subsample
            idx = np.arange(0, len(chunk_pts), subsample)
            points_list.append(chunk_pts[idx])
            total += len(chunk_pts)
            if len(points_list) % 10 == 0:
                kept = sum(len(p) for p in points_list)
                print(f"    Read {total/1e6:.1f}M points, kept {kept/1e3:.0f}K...")

    points = np.vstack(points_list)
    del points_list

    print(f"  Kept {len(points)} of {total} points ({100*len(points)/total:.1f}%)")
    print(f"  X range: [{points[:,0].min():.2f}, {points[:,0].max():.2f}]")
    print(f"  Y range: [{points[:,1].min():.2f}, {points[:,1].max():.2f}]")
    print(f"  Z range: [{points[:,2].min():.2f}, {points[:,2].max():.2f}]")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    del points
    return pcd


###############################################################################
# 2. Load SLAM map
###############################################################################

def load_slam_map(path):
    """Load PCD file from SLAM output."""
    print(f"[SLAM] Loading {path}...")
    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points)
    print(f"  {len(points)} points")
    print(f"  X range: [{points[:,0].min():.2f}, {points[:,0].max():.2f}]")
    print(f"  Y range: [{points[:,1].min():.2f}, {points[:,1].max():.2f}]")
    print(f"  Z range: [{points[:,2].min():.2f}, {points[:,2].max():.2f}]")
    return pcd


###############################################################################
# 3. Align clouds
###############################################################################

def center_cloud(pcd):
    """Subtract the centroid so the cloud is centered at origin."""
    pts = np.asarray(pcd.points)
    centroid = pts.mean(axis=0)
    pts_centered = pts - centroid
    pcd_centered = o3d.geometry.PointCloud()
    pcd_centered.points = o3d.utility.Vector3dVector(pts_centered)
    if pcd.has_colors():
        pcd_centered.colors = pcd.colors
    return pcd_centered, centroid


def align_clouds(gt_pcd, slam_pcd, voxel_size=0.5):
    """
    Align SLAM map to ground truth using:
      1. Center both clouds
      2. Downsample for speed
      3. Coarse alignment with RANSAC feature matching
      4. Fine alignment with ICP
    """
    print("\n[ALIGN] Aligning SLAM to ground truth...")

    # Center both
    gt_centered, gt_centroid = center_cloud(gt_pcd)
    slam_centered, slam_centroid = center_cloud(slam_pcd)
    print(f"  GT centroid:   {gt_centroid}")
    print(f"  SLAM centroid: {slam_centroid}")

    # Downsample
    gt_down = gt_centered.voxel_down_sample(voxel_size)
    slam_down = slam_centered.voxel_down_sample(voxel_size)
    print(f"  GT downsampled:   {len(gt_down.points)} pts")
    print(f"  SLAM downsampled: {len(slam_down.points)} pts")

    # Compute normals and FPFH features
    for pcd in [gt_down, slam_down]:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size * 3, max_nn=30))

    gt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        gt_down, o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 5, max_nn=100))
    slam_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        slam_down, o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 5, max_nn=100))

    # RANSAC global registration
    print("  Running RANSAC global alignment...")
    ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        slam_down, gt_down, slam_fpfh, gt_fpfh,
        mutual_filter=True,
        max_correspondence_distance=voxel_size * 2,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 2)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))

    print(f"  RANSAC fitness: {ransac_result.fitness:.4f}")

    # ICP fine alignment
    print("  Running ICP refinement...")
    slam_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 3, max_nn=30))
    gt_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 3, max_nn=30))

    icp_result = o3d.pipelines.registration.registration_icp(
        slam_down, gt_down,
        voxel_size * 1.5,
        ransac_result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))

    print(f"  ICP fitness: {icp_result.fitness:.4f}")
    print(f"  ICP RMSE:    {icp_result.inlier_rmse:.4f} m")

    T_align = icp_result.transformation
    return gt_centered, slam_centered, T_align


###############################################################################
# 4. Compute accuracy metrics
###############################################################################

def compute_metrics(gt_pcd, slam_pcd_aligned, max_dist=5.0):
    """
    Compute cloud-to-cloud distance from SLAM to ground truth.
    For each SLAM point, find the nearest GT point.
    """
    print("\n[METRICS] Computing cloud-to-cloud distances...")

    gt_tree = o3d.geometry.KDTreeFlann(gt_pcd)
    slam_pts = np.asarray(slam_pcd_aligned.points)

    distances = []
    for i in range(len(slam_pts)):
        [_, idx, dist2] = gt_tree.search_knn_vector_3d(slam_pts[i], 1)
        d = np.sqrt(dist2[0])
        if d < max_dist:  # ignore outliers
            distances.append(d)

    distances = np.array(distances)
    inlier_ratio = len(distances) / len(slam_pts)

    print(f"  Points compared: {len(distances)}/{len(slam_pts)} "
          f"({inlier_ratio*100:.1f}% within {max_dist}m)")
    print(f"  Mean distance:   {distances.mean():.4f} m")
    print(f"  Median distance: {np.median(distances):.4f} m")
    print(f"  Std distance:    {distances.std():.4f} m")
    print(f"  Max distance:    {distances.max():.4f} m")
    print(f"  90th percentile: {np.percentile(distances, 90):.4f} m")
    print(f"  95th percentile: {np.percentile(distances, 95):.4f} m")

    return distances


###############################################################################
# 5. Visualization
###############################################################################

def plot_distance_histogram(distances, output_dir):
    """Plot histogram of cloud-to-cloud distances."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(distances, bins=100, color='steelblue', edgecolor='none')
    ax.axvline(distances.mean(), color='red', linestyle='--', label=f'Mean: {distances.mean():.3f} m')
    ax.axvline(np.median(distances), color='orange', linestyle='--', label=f'Median: {np.median(distances):.3f} m')
    ax.set_xlabel('Distance to GT (m)')
    ax.set_ylabel('Count')
    ax.set_title('Cloud-to-Cloud Distance Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    sorted_d = np.sort(distances)
    cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
    ax.plot(sorted_d, cdf, 'b-', lw=2)
    ax.axhline(0.9, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(0.95, color='gray', linestyle=':', alpha=0.5)
    idx90 = np.searchsorted(cdf, 0.9)
    idx95 = np.searchsorted(cdf, 0.95)
    if idx90 < len(sorted_d):
        ax.axvline(sorted_d[idx90], color='orange', linestyle='--',
                   label=f'90%: {sorted_d[idx90]:.3f} m')
    if idx95 < len(sorted_d):
        ax.axvline(sorted_d[idx95], color='red', linestyle='--',
                   label=f'95%: {sorted_d[idx95]:.3f} m')
    ax.set_xlabel('Distance to GT (m)')
    ax.set_ylabel('CDF')
    ax.set_title('Cumulative Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "accuracy_histogram.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"[VIS] Saved {path}")
    plt.show()


def color_by_error(slam_pcd, gt_pcd, max_color_dist=1.0):
    """Color the SLAM cloud by distance to GT (blue=close, red=far)."""
    gt_tree = o3d.geometry.KDTreeFlann(gt_pcd)
    slam_pts = np.asarray(slam_pcd.points)
    colors = np.zeros((len(slam_pts), 3))

    for i in range(len(slam_pts)):
        [_, idx, dist2] = gt_tree.search_knn_vector_3d(slam_pts[i], 1)
        d = min(np.sqrt(dist2[0]), max_color_dist)
        t = d / max_color_dist  # 0 = perfect, 1 = max error
        colors[i] = [t, 0, 1 - t]  # red = far, blue = close

    slam_pcd.colors = o3d.utility.Vector3dVector(colors)
    return slam_pcd


def view_aligned(gt_pcd, slam_pcd_aligned):
    """View both clouds overlaid: GT in gray, SLAM in blue."""
    gt_vis = o3d.geometry.PointCloud(gt_pcd)
    gt_vis.paint_uniform_color([0.6, 0.6, 0.6])

    slam_vis = o3d.geometry.PointCloud(slam_pcd_aligned)
    slam_vis.paint_uniform_color([0.1, 0.4, 1.0])

    print("[VIS] Opening viewer (gray=GT, blue=SLAM)...")
    o3d.visualization.draw_geometries(
        [gt_vis, slam_vis],
        window_name="GT (gray) vs SLAM (blue)",
        width=1400, height=900)


###############################################################################
# 6. Main
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="Compare SLAM map vs ground truth LAS")
    parser.add_argument("--gt", required=True, help="Path to 3D_point_cloud_GT.las")
    parser.add_argument("--slam", default="slam_output/global_map.pcd",
                        help="Path to SLAM map PCD")
    parser.add_argument("--output", default="accuracy_output")
    parser.add_argument("--voxel", type=float, default=0.5,
                        help="Voxel size for alignment (default: 0.5)")
    parser.add_argument("--view", action="store_true",
                        help="Open 3D viewer after comparison")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load
    gt_pcd = load_las(args.gt)
    slam_pcd = load_slam_map(args.slam)

    # Align
    gt_centered, slam_centered, T_align = align_clouds(gt_pcd, slam_pcd, args.voxel)
    slam_aligned = slam_centered.transform(T_align)

    # Metrics
    distances = compute_metrics(gt_centered, slam_aligned)

    # Save metrics
    np.savetxt(os.path.join(args.output, "distances.txt"), distances,
               header="cloud_to_cloud_distance_meters", fmt="%.6f")

    # Plot
    plot_distance_histogram(distances, args.output)

    # View
    if args.view:
        view_aligned(gt_centered, slam_aligned)

    # Summary
    print(f"\n{'='*50}")
    print(f"ACCURACY SUMMARY")
    print(f"{'='*50}")
    print(f"  Mean error:   {distances.mean():.4f} m")
    print(f"  Median error: {np.median(distances):.4f} m")
    print(f"  RMSE:         {np.sqrt(np.mean(distances**2)):.4f} m")
    print(f"  90th pct:     {np.percentile(distances, 90):.4f} m")


if __name__ == "__main__":
    main()
