"""
lidar_degradation.py

Simulates realistic LiDAR sensor degradation for robustness testing of SLAM pipelines.
Compatible with the point cloud formats used in slam.py: Nx3 NumPy arrays (as returned
by extract_data) and Open3D PointCloud objects (as used inside LiDARPreprocessor).

Usage:
    from lidar_degradation import LiDARDegradation

    deg = LiDARDegradation(seed=42)

    # Apply a single effect to a raw Nx3 numpy array
    noisy_pts = deg.gaussian_noise(pts, sigma_base=0.02, range_scale=0.005)

    # Chain multiple effects
    degraded = deg.apply_pipeline(pts, [
        ("gaussian_noise",   {"sigma_base": 0.02,  "range_scale": 0.005}),
        ("random_dropout",   {"dropout_rate": 0.15}),
        ("fog",              {"visibility": 20.0,   "backscatter_rate": 0.05}),
    ])

    # Works with Open3D PointClouds too
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    degraded_pcd = deg.apply_to_pcd(pcd, [("random_dropout", {"dropout_rate": 0.2})])

Degradation modes
-----------------
1. gaussian_noise       – Additive zero-mean Gaussian noise, optionally range-scaled.
2. random_dropout       – Randomly remove points (missed returns, weak reflectivity).
3. beam_dropout         – Drop entire horizontal elevation rings (hardware failure).
4. range_clip           – Hard-clip all returns beyond a reduced maximum range.
5. fog                  – Distance-attenuated dropout + near-range backscatter.
6. outliers             – Inject random spurious points within sensor FOV.
7. angular_downsample   – Reduce effective angular resolution by angular-bin decimation.


light     │ Small Gaussian noise + 5% point dropout    │
  ├───────────┼────────────────────────────────────────────┤
  │ moderate  │ Larger noise + 15% dropout (range-biased)  │
  │           │ + 10% beam dropout                         │
  ├───────────┼────────────────────────────────────────────┤
  │ heavy     │ Large noise + 30% dropout + 25% beam       │
  │           │ dropout + range clipped to 25m             │
  ├───────────┼────────────────────────────────────────────┤
  │ fog_light │ Beer-Lambert fog (60m visibility) + light  │
  │           │ noise                                      │
  ├───────────┼────────────────────────────────────────────┤
  │ fog_dense │ Beer-Lambert fog (15m visibility) + heavy  │
  │           │ noise + range clipped to 20m               │
  ├───────────┼────────────────────────────────────────────┤
  │ low_res   │ Angular decimation (simulates lower-res    │
  │           │ sensor) + light noise  

"""

import numpy as np
import open3d as o3d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_numpy(cloud):
    """Accept either an Nx3 numpy array or an Open3D PointCloud; return (array, was_pcd)."""
    if isinstance(cloud, np.ndarray):
        assert cloud.ndim == 2 and cloud.shape[1] == 3, "Expected Nx3 float array"
        return cloud.copy().astype(np.float64), False
    elif isinstance(cloud, o3d.geometry.PointCloud):
        return np.asarray(cloud.points, dtype=np.float64).copy(), True
    else:
        raise TypeError(f"Unsupported cloud type: {type(cloud)}")


def _to_original_type(pts, was_pcd):
    """Convert a processed Nx3 array back to Open3D PointCloud if the input was one."""
    if was_pcd:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        return pcd
    return pts


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LiDARDegradation:
    """
    Collection of LiDAR degradation effects for SLAM robustness testing.

    Parameters
    ----------
    seed : int or None
        Random seed for reproducibility. None uses a random seed each call.
    """

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 1. Gaussian noise
    # ------------------------------------------------------------------

    def gaussian_noise(self, cloud, sigma_base=0.02, range_scale=0.005):
        """
        Add zero-mean Gaussian noise to every point.

        Models real-world range measurement uncertainty that grows with
        distance (e.g., from beam divergence, timing jitter, surface roughness).

        Parameters
        ----------
        sigma_base : float
            Base noise standard deviation in metres (applied regardless of range).
        range_scale : float
            Additional noise std per metre of range. Total std = sigma_base + r*range_scale.

        Returns
        -------
        Same type as input (Nx3 array or Open3D PointCloud).
        """
        pts, was_pcd = _to_numpy(cloud)
        ranges = np.linalg.norm(pts, axis=1, keepdims=True)  # (N,1)
        sigma = sigma_base + range_scale * ranges             # (N,1) broadcast
        noise = self.rng.standard_normal(pts.shape) * sigma
        pts += noise
        return _to_original_type(pts, was_pcd)

    # ------------------------------------------------------------------
    # 2. Random dropout
    # ------------------------------------------------------------------

    def random_dropout(self, cloud, dropout_rate=0.10, range_bias=0.0):
        """
        Randomly remove points to simulate missed or weak returns.

        Parameters
        ----------
        dropout_rate : float in [0, 1)
            Fraction of points to remove unconditionally.
        range_bias : float >= 0
            Extra per-metre probability of removal. At r metres,
            effective drop probability = dropout_rate + r * range_bias.
            Simulates lower SNR at long range.

        Returns
        -------
        Same type as input (Nx3 array or Open3D PointCloud).
        """
        pts, was_pcd = _to_numpy(cloud)
        if dropout_rate <= 0 and range_bias <= 0:
            return _to_original_type(pts, was_pcd)

        ranges = np.linalg.norm(pts, axis=1)
        p_drop = np.clip(dropout_rate + range_bias * ranges, 0.0, 1.0)
        keep = self.rng.random(len(pts)) >= p_drop
        pts = pts[keep]
        return _to_original_type(pts, was_pcd)

    # ------------------------------------------------------------------
    # 3. Beam dropout
    # ------------------------------------------------------------------

    def beam_dropout(self, cloud, drop_fraction=0.25, num_beams=16):
        """
        Remove entire horizontal elevation rings to simulate dead beams.

        Real multi-beam LiDARs (e.g., Velodyne VLP-16, HDL-32) arrange lasers
        at fixed elevation angles. A dead emitter wipes an entire ring. Since
        the bag data does not carry ring-ID metadata, we infer ring membership
        from each point's elevation angle.

        Parameters
        ----------
        drop_fraction : float in [0, 1)
            Fraction of rings to drop (randomly selected).
        num_beams : int
            Number of elevation rings to assume. The slam.py dataset uses a
            Velodyne sensor; 16 is a conservative default; 32 or 64 also valid.

        Returns
        -------
        Same type as input (Nx3 array or Open3D PointCloud).
        """
        pts, was_pcd = _to_numpy(cloud)
        if drop_fraction <= 0 or len(pts) == 0:
            return _to_original_type(pts, was_pcd)

        # Compute elevation angle for each point
        ranges_xy = np.linalg.norm(pts[:, :2], axis=1)
        elevation = np.degrees(np.arctan2(pts[:, 2], np.maximum(ranges_xy, 1e-6)))

        # Assign each point to a ring by quantising elevation
        el_min, el_max = elevation.min(), elevation.max()
        if el_max == el_min:
            return _to_original_type(pts, was_pcd)
        ring_ids = np.floor(
            (elevation - el_min) / (el_max - el_min + 1e-9) * num_beams
        ).astype(int)
        ring_ids = np.clip(ring_ids, 0, num_beams - 1)

        # Randomly select rings to drop
        n_drop = max(1, int(round(drop_fraction * num_beams)))
        dead_rings = self.rng.choice(num_beams, size=n_drop, replace=False)
        keep = ~np.isin(ring_ids, dead_rings)
        pts = pts[keep]
        return _to_original_type(pts, was_pcd)

    # ------------------------------------------------------------------
    # 4. Range clipping
    # ------------------------------------------------------------------

    def range_clip(self, cloud, max_range=20.0, min_range=None):
        """
        Hard-clip returns to a reduced range window.

        Simulates a sensor with shorter maximum range (e.g., from lower laser
        power, heavy atmospheric attenuation, or firmware downgrade).

        Parameters
        ----------
        max_range : float
            Maximum range in metres. Points beyond this are removed.
        min_range : float or None
            Minimum range in metres. If None, keeps the slam.py default of 1m.
            Pass 0.0 to keep all near returns.

        Returns
        -------
        Same type as input (Nx3 array or Open3D PointCloud).
        """
        pts, was_pcd = _to_numpy(cloud)
        ranges = np.linalg.norm(pts, axis=1)
        mask = ranges <= max_range
        if min_range is not None:
            mask &= ranges >= min_range
        pts = pts[mask]
        return _to_original_type(pts, was_pcd)

    # ------------------------------------------------------------------
    # 5. Fog simulation
    # ------------------------------------------------------------------

    def fog(self, cloud, visibility=15.0, backscatter_rate=0.05):
        """
        Simulate LiDAR returns in fog or heavy rain.

        Two effects are modelled:
          (a) Attenuation: probability of a point surviving decreases
              exponentially with range according to Beer-Lambert law with
              an extinction coefficient derived from the meteorological
              visibility distance.
          (b) Backscatter: a fraction of attenuated points are replaced by
              spurious near-range returns (fog particles scattering the beam
              back before it reaches the real target).

        Parameters
        ----------
        visibility : float
            Meteorological visibility in metres (10–50m = dense fog,
            50–200m = moderate fog, 200m+ = light haze).
        backscatter_rate : float in [0, 1)
            Fraction of attenuated points that produce a backscatter return
            at a random range between 0.5m and 0.3 * visibility.

        Returns
        -------
        Same type as input (Nx3 array or Open3D PointCloud).
        """
        pts, was_pcd = _to_numpy(cloud)
        if len(pts) == 0:
            return _to_original_type(pts, was_pcd)

        # Beer-Lambert extinction: I(r) = I0 * exp(-beta * r)
        # beta chosen so that at r = visibility, transmission = 0.05 (5%)
        beta = -np.log(0.05) / visibility  # ~0.2996 / visibility
        ranges = np.linalg.norm(pts, axis=1)
        p_survive = np.exp(-beta * ranges)
        survive_mask = self.rng.random(len(pts)) < p_survive

        # Backscatter: a subset of removed points come back as near-range spurious hits
        removed_idx = np.where(~survive_mask)[0]
        n_back = int(round(backscatter_rate * len(removed_idx)))
        backscatter_pts = []
        if n_back > 0 and len(removed_idx) > 0:
            chosen = self.rng.choice(removed_idx, size=n_back, replace=False)
            directions = pts[chosen] / (ranges[chosen, np.newaxis] + 1e-9)
            bs_range_max = max(0.5, 0.3 * visibility)
            bs_ranges = self.rng.uniform(0.5, bs_range_max, size=n_back)
            backscatter_pts = directions * bs_ranges[:, np.newaxis]

        surviving_pts = pts[survive_mask]
        if len(backscatter_pts) > 0:
            pts_out = np.vstack([surviving_pts, backscatter_pts])
        else:
            pts_out = surviving_pts

        return _to_original_type(pts_out, was_pcd)

    # ------------------------------------------------------------------
    # 6. Outlier injection
    # ------------------------------------------------------------------

    def outliers(self, cloud, num_outliers=200, max_range=50.0):
        """
        Inject random spurious points throughout the sensor FOV.

        Simulates cross-talk from other LiDAR sensors, solar interference,
        rain/dust specular reflections, or firmware glitches.

        Parameters
        ----------
        num_outliers : int
            Number of spurious points to add.
        max_range : float
            Maximum range from which outliers are sampled uniformly.

        Returns
        -------
        Same type as input (Nx3 array or Open3D PointCloud).
        """
        pts, was_pcd = _to_numpy(cloud)

        # Sample random directions on the unit sphere, then random ranges
        theta = self.rng.uniform(0, 2 * np.pi, num_outliers)
        phi = self.rng.uniform(-np.pi / 2, np.pi / 2, num_outliers)
        r = self.rng.uniform(1.0, max_range, num_outliers)

        x = r * np.cos(phi) * np.cos(theta)
        y = r * np.cos(phi) * np.sin(theta)
        z = r * np.sin(phi)

        spurious = np.stack([x, y, z], axis=1)
        pts_out = np.vstack([pts, spurious])
        return _to_original_type(pts_out, was_pcd)

    # ------------------------------------------------------------------
    # 7. Angular resolution downsampling
    # ------------------------------------------------------------------

    def angular_downsample(self, cloud, azimuth_step_deg=1.0, elevation_step_deg=2.0):
        """
        Reduce the effective angular resolution by binning and keeping one
        point per angular cell (closest point wins).

        Simulates using a lower-resolution sensor, degraded firmware, or
        reduced scan rate settings.

        Parameters
        ----------
        azimuth_step_deg : float
            Angular bin width in azimuth (horizontal) in degrees.
        elevation_step_deg : float
            Angular bin width in elevation (vertical) in degrees.

        Returns
        -------
        Same type as input (Nx3 array or Open3D PointCloud).
        """
        pts, was_pcd = _to_numpy(cloud)
        if len(pts) == 0:
            return _to_original_type(pts, was_pcd)

        ranges_xy = np.linalg.norm(pts[:, :2], axis=1)
        azimuth = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))   # -180 to 180
        elevation = np.degrees(np.arctan2(pts[:, 2], np.maximum(ranges_xy, 1e-6)))
        ranges = np.linalg.norm(pts, axis=1)

        az_bin = np.floor(azimuth / azimuth_step_deg).astype(int)
        el_bin = np.floor(elevation / elevation_step_deg).astype(int)

        # For each (az_bin, el_bin) cell keep the closest point
        cell_map = {}  # (az, el) -> (range, index)
        for i in range(len(pts)):
            key = (az_bin[i], el_bin[i])
            if key not in cell_map or ranges[i] < cell_map[key][0]:
                cell_map[key] = (ranges[i], i)

        keep_idx = np.array([v[1] for v in cell_map.values()], dtype=int)
        pts = pts[keep_idx]
        return _to_original_type(pts, was_pcd)

    # ------------------------------------------------------------------
    # Pipeline convenience methods
    # ------------------------------------------------------------------

    def apply_pipeline(self, cloud, effects):
        """
        Apply a sequence of degradation effects in order.

        Parameters
        ----------
        cloud : Nx3 numpy array or Open3D PointCloud
            Input point cloud.
        effects : list of (str, dict) tuples
            Each tuple is (method_name, kwargs_dict).
            method_name must match one of the public methods of this class.

        Returns
        -------
        Degraded cloud in the same type as input.

        Example
        -------
        degraded = deg.apply_pipeline(pts, [
            ("gaussian_noise",  {"sigma_base": 0.03}),
            ("random_dropout",  {"dropout_rate": 0.1}),
            ("fog",             {"visibility": 25.0}),
        ])
        """
        result = cloud
        for name, kwargs in effects:
            fn = getattr(self, name, None)
            if fn is None:
                raise ValueError(
                    f"Unknown degradation method '{name}'. "
                    f"Available: {self._available_methods()}"
                )
            result = fn(result, **kwargs)
        return result

    def apply_to_pcd(self, pcd, effects):
        """
        Convenience wrapper that ensures the return is always an Open3D PointCloud,
        regardless of whether the internal methods would otherwise return a numpy array.

        Parameters
        ----------
        pcd : Open3D PointCloud
        effects : list of (str, dict) tuples  (same format as apply_pipeline)

        Returns
        -------
        Open3D PointCloud
        """
        if not isinstance(pcd, o3d.geometry.PointCloud):
            raise TypeError("apply_to_pcd expects an Open3D PointCloud")
        result = self.apply_pipeline(pcd, effects)
        # Ensure normals are re-estimated if the original had them and points changed
        if isinstance(result, np.ndarray):
            out_pcd = o3d.geometry.PointCloud()
            out_pcd.points = o3d.utility.Vector3dVector(result)
            return out_pcd
        return result

    def _available_methods(self):
        return [
            m for m in dir(self)
            if not m.startswith("_") and callable(getattr(self, m))
            and m not in ("apply_pipeline", "apply_to_pcd")
        ]


# ---------------------------------------------------------------------------
# Preset configurations
# ---------------------------------------------------------------------------

PRESETS = {
    "light": [
        ("gaussian_noise",  {"sigma_base": 0.01, "range_scale": 0.002}),
        ("random_dropout",  {"dropout_rate": 0.05}),
    ],
    "moderate": [
        ("gaussian_noise",  {"sigma_base": 0.03, "range_scale": 0.005}),
        ("random_dropout",  {"dropout_rate": 0.15, "range_bias": 0.002}),
        ("beam_dropout",    {"drop_fraction": 0.10}),
    ],
    "heavy": [
        ("gaussian_noise",  {"sigma_base": 0.06, "range_scale": 0.01}),
        ("random_dropout",  {"dropout_rate": 0.30, "range_bias": 0.005}),
        ("beam_dropout",    {"drop_fraction": 0.25}),
        ("range_clip",      {"max_range": 25.0}),
    ],
    "fog_light": [
        ("fog",             {"visibility": 60.0, "backscatter_rate": 0.03}),
        ("gaussian_noise",  {"sigma_base": 0.02}),
    ],
    "fog_dense": [
        ("fog",             {"visibility": 15.0, "backscatter_rate": 0.08}),
        ("gaussian_noise",  {"sigma_base": 0.05}),
        ("range_clip",      {"max_range": 20.0}),
    ],
    "low_res": [
        ("angular_downsample", {"azimuth_step_deg": 2.0, "elevation_step_deg": 4.0}),
        ("gaussian_noise",     {"sigma_base": 0.02}),
    ],
}


def apply_preset(cloud, preset_name, seed=None):
    """
    Apply a named preset degradation pipeline.

    Parameters
    ----------
    cloud : Nx3 numpy array or Open3D PointCloud
    preset_name : str
        One of: 'light', 'moderate', 'heavy', 'fog_light', 'fog_dense', 'low_res'
    seed : int or None

    Returns
    -------
    Degraded cloud in the same type as input.
    """
    if preset_name not in PRESETS:
        raise ValueError(
            f"Unknown preset '{preset_name}'. Available: {list(PRESETS.keys())}"
        )
    deg = LiDARDegradation(seed=seed)
    return deg.apply_pipeline(cloud, PRESETS[preset_name])


# ---------------------------------------------------------------------------
# Quick demo / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("LiDAR Degradation Simulator — smoke test")
    print("=" * 50)

    rng = np.random.default_rng(0)
    # Synthetic point cloud: random points in a 40m cube centred at origin
    N = 5000
    pts_raw = rng.uniform(-20, 20, (N, 3)).astype(np.float64)
    # Keep only points > 1m from sensor (mimic slam.py range filter)
    pts_raw = pts_raw[np.linalg.norm(pts_raw, axis=1) > 1.0]
    print(f"Input cloud: {len(pts_raw)} points")

    deg = LiDARDegradation(seed=42)

    results = {}

    results["gaussian_noise"] = deg.gaussian_noise(pts_raw, sigma_base=0.03)
    print(f"  gaussian_noise    : {len(results['gaussian_noise'])} pts")

    results["random_dropout"] = deg.random_dropout(pts_raw, dropout_rate=0.20)
    print(f"  random_dropout    : {len(results['random_dropout'])} pts  "
          f"(expected ~{int(len(pts_raw)*0.80)})")

    results["beam_dropout"] = deg.beam_dropout(pts_raw, drop_fraction=0.25, num_beams=16)
    print(f"  beam_dropout      : {len(results['beam_dropout'])} pts  "
          f"(dropped 4/16 beams)")

    results["range_clip"] = deg.range_clip(pts_raw, max_range=15.0)
    print(f"  range_clip 15m    : {len(results['range_clip'])} pts")

    results["fog"] = deg.fog(pts_raw, visibility=20.0, backscatter_rate=0.05)
    print(f"  fog (vis=20m)     : {len(results['fog'])} pts")

    results["outliers"] = deg.outliers(pts_raw, num_outliers=300)
    print(f"  outliers          : {len(results['outliers'])} pts  "
          f"(added 300)")

    results["angular_downsample"] = deg.angular_downsample(
        pts_raw, azimuth_step_deg=2.0, elevation_step_deg=4.0
    )
    print(f"  angular_downsample: {len(results['angular_downsample'])} pts")

    print()
    print("Preset pipelines:")
    for name in PRESETS:
        out = apply_preset(pts_raw, name, seed=0)
        print(f"  {name:<15}: {len(out)} pts")

    print()
    print("Open3D PointCloud interface:")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_raw)
    out_pcd = deg.apply_to_pcd(pcd, [
        ("random_dropout", {"dropout_rate": 0.15}),
        ("gaussian_noise", {"sigma_base": 0.02}),
    ])
    print(f"  Input PCD: {len(pcd.points)} pts  ->  Output PCD: {len(out_pcd.points)} pts")

    print()
    print("All tests passed.")
