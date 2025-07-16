
import open3d as o3d
import numpy as np
import json

def load_scannet_ply_with_labels(ply_file_path, label_ply_file_path):
    try:
        # Load the mesh (point cloud with colors)
        pcd = o3d.io.read_point_cloud(ply_file_path)
        if not pcd.has_points():
            print(f"Warning: No points found in {ply_file_path}")
            return None

        # Load the labels
        label_pcd = o3d.io.read_point_cloud(label_ply_file_path)
        if not label_pcd.has_points():
            print(f"Warning: No points found in label file {label_ply_file_path}")
            return None
        
        # Ensure the number of points match (simple check, more robust check needed for real data)
        if len(pcd.points) != len(label_pcd.points):
            print(f"Error: Point count mismatch between mesh ({len(pcd.points)}) and labels ({len(label_pcd.points)}).")
            return None

        
        if label_pcd.has_colors():
            # Assuming labels are encoded in the red channel of colors (0-255)
            labels = np.asarray(label_pcd.colors)[:, 0] * 255 # Scale to 0-255 if normalized
            labels = labels.astype(np.int32)
        else:
            print("Warning: Label PLY does not have colors. Cannot extract labels easily.")
            labels = np.zeros(len(pcd.points), dtype=np.int32) # Dummy labels

        print(f"Successfully loaded {len(pcd.points)} points and labels from {ply_file_path} and {label_ply_file_path}")
        return pcd, labels
    except Exception as e:
        print(f"Error loading PLY files: {e}")
        return None, None

def load_scannet_labels_from_json(aggregation_json_path, segs_json_file_path, num_points):
    
    try:
        with open(aggregation_json_path, 'r') as f: 
            aggregation_data = json.load(f)
        with open(segs_json_file_path, 'r') as f:
            segs_data = json.load(f)

        # Initialize labels array with a default value (e.g., -1 for unassigned)
        labels = np.full(num_points, -1, dtype=np.int32)

        # segIndices maps vertex index to segment ID
        seg_indices = np.array(segs_data["segIndices"])

        # Create a mapping from string labels to integer IDs
        unique_labels = sorted(list(set([sg["label"] for sg in aggregation_data["segGroups"]])))
        label_to_id = {label: i for i, label in enumerate(unique_labels)}

        # segGroups maps segment ID to semantic label
        for seg_group in aggregation_data["segGroups"]:
            string_label = seg_group["label"]
            label_id = label_to_id.get(string_label, -1) # Get integer ID, -1 if not found
            segments = seg_group["segments"]
            for seg_id in segments:
                # Find all vertex indices belonging to this segment ID
                vertex_indices = np.where(seg_indices == seg_id)[0]
                labels[vertex_indices] = label_id
        
        print(f"Successfully extracted labels from JSON files for {num_points} points.")
        return labels, label_to_id

    except Exception as e:
        print(f"Error loading labels from JSON files: {e}")
        return None, None


if __name__ == "__main__":
    # Paths to the downloaded files for scene0000_00
    mesh_file_path = "scannet_data/scans/scene0000_00/scene0000_00_vh_clean_2.ply"
    label_ply_file_path = "scannet_data/scans/scene0000_00/scene0000_00_vh_clean_2.labels.ply"
    aggregation_json_path = "scannet_data/scans/scene0000_00/scene0000_00_vh_clean.aggregation.json"
    segs_json_path = "scannet_data/scans/scene0000_00/scene0000_00_vh_clean_2.0.010000.segs.json"

    # Load point cloud and labels from PLY files
    point_cloud_ply, labels_ply = load_scannet_ply_with_labels(mesh_file_path, label_ply_file_path)

    if point_cloud_ply is not None and labels_ply is not None:
        print(f"Labels from PLY (first 10): {labels_ply[:10]}")
        print(f"Unique labels from PLY: {np.unique(labels_ply)}")

    # Load labels from JSON files (more accurate for semantic segmentation)
    # First, load the mesh to get the number of points
    temp_pcd = o3d.io.read_point_cloud(mesh_file_path)
    if temp_pcd is not None:
        num_points_in_mesh = len(temp_pcd.points)
        labels_json, label_map = load_scannet_labels_from_json(aggregation_json_path, segs_json_path, num_points_in_mesh)

        if labels_json is not None:
            print(f"Labels from JSON (first 10): {labels_json[:10]}")
            print(f"Unique labels from JSON: {np.unique(labels_json)}")
            print(f"Label to ID mapping: {label_map}")

        

    print("Data preparation script finished.")


