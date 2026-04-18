# SLAM with LiDAR and Radar Fusion

Multi-mode SLAM implementation comparing LiDAR-only, radar fusion, and degradation scenarios.

## slam.py — Multi-Mode SLAM

Main SLAM pipeline with 7 modes. Outputs trajectories and maps to `slam_output/<mode>/`.

**Usage:**
```bash
python slam.py --bag <path_to_bag> [--mode MODE] [--output DIR]
```

**Arguments:**
- `--bag` (required): Path to ROS bag file
- `--mode`: One or more modes, or 'all' (default: baseline)
- `--output`: Output directory (default: slam_output)

**Modes:**
- `baseline`: LiDAR + odometry (reference)
- `standard_degraded`: LiDAR with mild fog/dust simulation + odometry
- `heavy_degraded`: LiDAR with heavy fog/dust simulation + odometry
- `radar`: LiDAR + radar fusion (30% radar weight)
- `standard_degraded_radar`: Standard degraded + radar (70% radar weight)
- `heavy_degraded_radar`: Heavy degraded + radar (90% radar weight)
- `radar_only`: Radar-only pose estimation with LiDAR mapping
- `odom_only`: Pure odometry (no SLAM correction)

**Examples:**
```bash
python slam.py --bag data.bag --mode baseline
python slam.py --bag data.bag --mode all
python slam.py --bag data.bag --mode standard_degraded heavy_degraded_radar
```

**Outputs:** `slam_output/<mode>/`
- `trajectory.txt`: Pose estimates
- `global_map.pcd`: 3D point cloud map
- `fitness_log.txt`: ICP fitness values
- `trajectory_2d.png`, `trajectory_3d.png`: Trajectory plots
- `fusion_diagnostics.png`: Radar/LiDAR fitness and fusion weights (radar modes only)

---

## radar_slam.py — Radar-Only SLAM

Pure radar SLAM with motion compensation and outlier filtering. Outputs to `radar_slam_output/`.

**Usage:**
```bash
python radar_slam.py --bag <path_to_bag> [--output DIR]
```

**Arguments:**
- `--bag` (required): Path to ROS bag file
- `--output`: Output directory (default: radar_slam_output)

**Outputs:** `radar_slam_output/`
- `trajectory.txt`: Radar-only trajectory
- `global_map.pcd`: 2D radar point cloud
- `radar_slam.png`: Combined map + SLAM/odometry trajectories

---

## compare.py — Compare All Results

Auto-detects and plots all modes from output directory. Generates trajectory comparisons and accuracy vs ground truth (if GT cache available).

**Usage:**
```bash
python compare.py [--dir DIR] [--gt PATH_TO_GT.las]
```

**Arguments:**
- `--dir`: Input directory containing mode results (default: slam_output)
- `--gt`: Path to ground truth LAS file (first time only, creates cache)

**Outputs:** `slam_output/`
- `comparison_2d.png`: 2D trajectories (bird's eye)
- `comparison_3d.png`: 3D trajectories
- `comparison_height.png`: Height profiles over time
- `comparison_fitness.png`: ICP fitness over time
- `accuracy_comparison.png`: Mean/median/RMSE/90th %ile error vs GT
- `gt_cache.pcd`: Cached ground truth (created on first run)

**Examples:**
```bash
python compare.py
python compare.py --gt ~/datasets/GT.las  # First time with GT
python compare.py --dir slam_output       # Repeat runs (uses cached GT)
```

---

## view_map.py — Interactive 3D Map Viewer

Visualize point cloud maps and trajectories. Automatically aligns GT to first map if requested.

**Usage:**
```bash
python view_map.py <dir1> [<dir2> ...] [--gt]
```

**Arguments:**
- `dirs`: One or more SLAM output directories
- `--gt`: Overlay ground truth (requires cached GT from compare.py)

**Examples:**
```bash
python view_map.py slam_output/baseline
python view_map.py slam_output/baseline slam_output/standard_degraded --gt
python view_map.py slam_output/standard_degraded slam_output/standard_degraded_radar --gt
```

---

## Workflow

1. **Run SLAM:**
   ```bash
   python slam.py --bag data.bag --mode all
   ```

2. **Compare results (first time with GT):**
   ```bash
   python compare.py --gt ~/datasets/GT.las
   ```

3. **Compare results (subsequent runs):**
   ```bash
   python compare.py
   ```

4. **Inspect maps:**
   ```bash
   python view_map.py slam_output/baseline slam_output/heavy_degraded_radar --gt
   ```

5. **Radar-only comparison (optional):**
   ```bash
   python radar_slam.py --bag data.bag
   ```
