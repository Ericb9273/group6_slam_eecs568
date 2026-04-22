
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt

COLORS = {
    'baseline':                 ('tab:blue',   'Baseline (LiDAR)'),
    'standard_degraded':        ('tab:orange', 'Standard Degraded'),
    'heavy_degraded':           ('tab:red',    'Heavy Degraded'),
    'radar':                    ('tab:green',  'LiDAR + Radar'),
    'standard_degraded_radar':  ('tab:cyan',   'Standard Degraded + Radar'),
    'heavy_degraded_radar':     ('tab:purple', 'Heavy Degraded + Radar'),
    'radar_only':               ('goldenrod',  'Radar Only'),
    'odom_only':                ('tab:gray',   'Odom Only'),
}

GROUP1 = ['odom_only', 'radar_only', 'baseline', 'radar']
GROUP2 = ['standard_degraded', 'standard_degraded_radar', 'heavy_degraded', 'heavy_degraded_radar']


def load_gt_results(slam_dir):
    """Load precomputed gt_results from accuracy_comparison data via trajectory files."""
    import json
    cache = os.path.join(slam_dir, "gt_results.json")
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)
    return None


def plot_group(ax, modes, gt_results, title):
    present = [m for m in modes if m in gt_results]
    x = np.arange(len(present))
    width = 0.2

    bars1 = ax.bar(x - 1.5*width, [gt_results[m]['mean']   for m in present], width, label='Mean',   color='steelblue')
    bars2 = ax.bar(x - 0.5*width, [gt_results[m]['median'] for m in present], width, label='Median', color='orange')
    bars3 = ax.bar(x + 0.5*width, [gt_results[m]['rmse']   for m in present], width, label='RMSE',   color='green')
    bars4 = ax.bar(x + 1.5*width, [gt_results[m]['p90']    for m in present], width, label='90th percentile', color='red')

    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=7)

    labels = [COLORS.get(m, ('', m))[1] for m in present]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('Distance to GT (m)')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="slam_output")
    args = parser.parse_args()

    gt_results = load_gt_results(args.dir)
    if gt_results is None:
        print("No gt_results.json found. Run compare.py first, then save gt_results.")
        print("Alternatively, hardcode your results below.")

        # --- Hardcoded fallback from your actual results ---
        gt_results = {
            'odom_only':               {'mean': 0.449, 'median': 0.292, 'rmse': 0.699, 'p90': 0.889},
            'radar_only':              {'mean': 0.579, 'median': 0.354, 'rmse': 0.876, 'p90': 1.277},
            'baseline':                {'mean': 0.311, 'median': 0.185, 'rmse': 0.616, 'p90': 0.423},
            'radar':                   {'mean': 0.332, 'median': 0.206, 'rmse': 0.615, 'p90': 0.474},
            'standard_degraded':       {'mean': 0.631, 'median': 0.303, 'rmse': 1.037, 'p90': 1.717},
            'standard_degraded_radar': {'mean': 0.404, 'median': 0.264, 'rmse': 0.672, 'p90': 0.692},
            'heavy_degraded':          {'mean': 1.015, 'median': 0.449, 'rmse': 1.524, 'p90': 2.768},
            'heavy_degraded_radar':    {'mean': 0.394, 'median': 0.252, 'rmse': 0.656, 'p90': 0.680},
        }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

    plot_group(ax1, GROUP1, gt_results, 'Baseline Configurations')
    plot_group(ax2, GROUP2, gt_results, 'Degradation Configurations')

    fig.suptitle('SLAM Map Accuracy vs Ground Truth', fontsize=13, fontweight='bold')
    plt.tight_layout()

    out = os.path.join(args.dir, "accuracy_comparison_split.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
