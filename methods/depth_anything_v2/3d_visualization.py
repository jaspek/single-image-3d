from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from si3d.datasets.nyu import NYUDepthV2
import open3d as o3d
import argparse

CURRENT_DIR = Path(__file__).resolve().parent.parent
ROOT = CURRENT_DIR.parent

parser = argparse.ArgumentParser()
parser.add_argument("--path", required=False, type=str, default=f"{ROOT}/outputs/depth_anything_v2/nyu/0001.npy")
args = parser.parse_args()

filePath = Path(args.path)


# Load the depth map
depth = np.load(filePath)

# Check statistics
print("Shape:", depth.shape) 
print("Min distance:", depth.min(), "m")
print("Max distance:", depth.max(), "m")
print("Mean distance:", depth.mean(), "m")

# 3. Visualize as a colormap
plt.figure(figsize=(8, 6))
plt.imshow(depth, cmap='inferno')
plt.colorbar(label='Distance (meters)')
plt.title('Predicted Metric Depth Map')
plt.axis('off')
plt.show()

ds = NYUDepthV2("data/nyu")
#Load the corresponding RGB frame for point cloud texturing
rgb_image = ds.image(int(filePath.stem))

#Extract intrinsic parameters (NYUv2 defaults)
K = NYUDepthV2.intrinsics()

fx, fy = K[0, 0], K[1, 1]
cx, cy = K[0, 2], K[1, 2]

#Create pixel coordinate grid (u = columns, v = rows)
H, W = depth.shape
u_coords, v_coords = np.meshgrid(np.arange(W), np.arange(H))

# Filter out invalid or extreme depth values
mask = (depth > 0.1) & (depth < 10.0)

Z = depth[mask]
X = (u_coords[mask] - cx) * Z / fx
Y = (v_coords[mask] - cy) * Z / fy

# Stack into (N, 3) 3D Point Cloud matrix
points_3d = np.stack((X, Y, Z), axis=-1)

# Extract matching RGB colors and normalize to [0, 1] range for Open3D
colors_rgb = rgb_image[mask] / 255.0

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_3d)
pcd.colors = o3d.utility.Vector3dVector(colors_rgb)

# Invert Y axis because image coordinate origin (0,0) is top-left, 
# whereas standard 3D camera coordinates treat Y as UP.
pcd.transform([[1, 0, 0, 0], 
               [0, -1, 0, 0], 
               [0, 0, -1, 0], 
               [0, 0, 0, 1]])

o3d.visualization.draw_geometries([pcd])