
import open3d as o3d
import numpy as np

def load_scannet_ply(file_path):
    """
    Loads a PLY file from ScanNet dataset and returns an Open3D PointCloud object.
    """
    try:
        pcd = o3d.io.read_point_cloud(file_path)
        if not pcd.has_points():
            print(f"Warning: No points found in {file_path}")
            return None
        print(f"Successfully loaded {len(pcd.points)} points from {file_path}")
        return pcd
    except Exception as e:
        print(f"Error loading PLY file {file_path}: {e}")
        return None

def visualize_point_cloud(pcd):
    """
    Visualizes an Open3D PointCloud object.
    """
    if pcd:
        o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    # Path to the downloaded PLY file for scene0000_00
    # We'll use the cleaned and decimated mesh for semantic annotations
    ply_file_path = "scannet_data/scans/scene0000_00/scene0000_00_vh_clean_2.ply"

    # Load the point cloud
    point_cloud = load_scannet_ply(ply_file_path)

    # Visualize the point cloud (this will open a new window if run locally)
    # In a headless environment like this sandbox, direct visualization might not work.
    # We will verify by checking if the point cloud object is not None and has points.
    if point_cloud:
        print("Point cloud loaded successfully. Visualization attempt (may not display in sandbox).")
        # For sandbox environment, we can check properties instead of drawing
        print(f"Number of points: {len(point_cloud.points)}")
        print(f"Has colors: {point_cloud.has_colors()}")
        print(f"Has normals: {point_cloud.has_normals()}")



