#!/usr/bin/env python3
"""
run_with_degradation.py

Wrapper around slam.py that applies LiDAR degradation before the SLAM pipeline.
Edit the CONFIG section below to change the bag path, preset, and output settings.

Usage:
    python run_with_degradation.py
    python run_with_degradation.py --preset fog_dense
    python run_with_degradation.py --preset moderate --max-scans 300
    python run_with_degradation.py --no-vis
"""

import argparse
import os

from slam import (
    extract_data,
    get_calibrated_lidar_to_base,
    run_slam,
    save_results,
    visualize_all,
)
from lidar_degradation import LiDARDegradation, apply_preset, PRESETS


###############################################################################
# CONFIG — edit these defaults
###############################################################################

DEFAULT_BAG     = "~/datasets/03_80m_other_sensor.bag"
DEFAULT_PRESET  = "moderate"
DEFAULT_OUTPUT  = "slam_output_degraded"
DEFAULT_SEED    = 42


###############################################################################
# MAIN
###############################################################################

def main():
    parser = argparse.ArgumentParser(
        description="Run SLAM with simulated LiDAR degradation"
    )
    parser.add_argument("--bag",       default=DEFAULT_BAG,
                        help="Path to ROS bag file")
    parser.add_argument("--preset",    default=DEFAULT_PRESET,
                        choices=list(PRESETS.keys()),
                        help=f"Degradation preset (default: {DEFAULT_PRESET})")
    parser.add_argument("--max-scans", type=int, default=None,
                        help="Limit number of LiDAR scans processed")
    parser.add_argument("--output",    default=None,
                        help="Output directory (default: slam_output_<preset>)")
    parser.add_argument("--submap-size", type=int, default=20,
                        help="Local submap size (default: 20)")
    parser.add_argument("--seed",      type=int, default=DEFAULT_SEED,
                        help="Random seed for degradation (default: 42)")
    parser.add_argument("--no-vis",    action="store_true",
                        help="Skip trajectory visualisation plots")
    parser.add_argument("--no-degradation", action="store_true",
                        help="Run clean SLAM (no degradation) for baseline comparison")
    args = parser.parse_args()

    output_dir = args.output or f"slam_output_{args.preset}"
    if args.no_degradation:
        output_dir = "slam_output_clean"

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    lidar_data, odom_data = extract_data(args.bag, args.max_scans)

    # ------------------------------------------------------------------
    # 2. Apply degradation
    # ------------------------------------------------------------------
    if args.no_degradation:
        print(f"\n[DEG] Degradation disabled — running clean baseline")
        degraded_lidar = lidar_data
    else:
        print(f"\n[DEG] Applying preset '{args.preset}' to {len(lidar_data)} scans "
              f"(seed={args.seed})...")
        print(f"[DEG] Effects:")
        for name, kwargs in PRESETS[args.preset]:
            print(f"        {name}({', '.join(f'{k}={v}' for k, v in kwargs.items())})")

        deg = LiDARDegradation(seed=args.seed)
        degraded_lidar = []
        pts_before = 0
        pts_after  = 0

        for stamp, pts in lidar_data:
            pts_deg = deg.apply_pipeline(pts, PRESETS[args.preset])
            degraded_lidar.append((stamp, pts_deg))
            pts_before += len(pts)
            pts_after  += len(pts_deg)

        avg_before = pts_before / len(lidar_data)
        avg_after  = pts_after  / len(degraded_lidar)
        print(f"[DEG] Avg points per scan: {avg_before:.0f} -> {avg_after:.0f} "
              f"({100*avg_after/avg_before:.1f}% retained)")

    # ------------------------------------------------------------------
    # 3. Run SLAM
    # ------------------------------------------------------------------
    T_l2b = get_calibrated_lidar_to_base()
    trajectory, map_pcds, closures = run_slam(
        degraded_lidar, odom_data, T_l2b,
        submap_size=args.submap_size,
    )

    # ------------------------------------------------------------------
    # 4. Save
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    traj_arr = save_results(trajectory, map_pcds, output_dir)

    # ------------------------------------------------------------------
    # 5. Visualise
    # ------------------------------------------------------------------
    if not args.no_vis:
        visualize_all(traj_arr, odom_data, closures, output_dir)

    # ------------------------------------------------------------------
    # 6. Summary
    # ------------------------------------------------------------------
    import numpy as np
    d = np.linalg.norm(traj_arr[-1, 1:4] - traj_arr[0, 1:4])
    print(f"\n[SUMMARY]")
    print(f"  Preset:            {args.preset if not args.no_degradation else 'none (clean)'}")
    print(f"  Output dir:        {output_dir}/")
    print(f"  SLAM displacement: {d:.2f} m")
    print(f"  Loop closures:     {len(closures)}")


if __name__ == "__main__":
    main()
