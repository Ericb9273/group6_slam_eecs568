# Add to check_radar.py
import rosbag
import numpy as np
import os

bag_path = os.path.expanduser('~/datasets/03_80m_other_sensor.bag')
bag = rosbag.Bag(bag_path)
import sensor_msgs.point_cloud2 as pc2

pts_list = []
for i, (topic, msg, t) in enumerate(bag.read_messages(topics=['/isdr_driver_1/points'])):
    pts = np.array(list(pc2.read_points(msg, field_names=("x","y","z","angle","intensity"), skip_nans=True)), dtype=np.float32)
    if len(pts) > 0:
        pts_list.append(pts[0])  # 1 point per msg
    if i >= 500:
        break
bag.close()

pts = np.array(pts_list)
print(f"Shape: {pts.shape}")
print(f"X   range: [{pts[:,0].min():.3f}, {pts[:,0].max():.3f}]")
print(f"Y   range: [{pts[:,1].min():.3f}, {pts[:,1].max():.3f}]")
print(f"Z   range: [{pts[:,2].min():.3f}, {pts[:,2].max():.3f}]")
print(f"Angle range: [{pts[:,3].min():.3f}, {pts[:,3].max():.3f}]")
print(f"Range (sqrt(x²+y²+z²)): [{np.linalg.norm(pts[:,:3],axis=1).min():.3f}, {np.linalg.norm(pts[:,:3],axis=1).max():.3f}]")
print(f"\nFirst 10 points:")
for p in pts[:10]:
    r = np.linalg.norm(p[:3])
    print(f"  x={p[0]:7.3f} y={p[1]:7.3f} z={p[2]:7.3f} angle={p[3]:7.3f} intensity={p[4]:7.1f} range={r:.3f}")