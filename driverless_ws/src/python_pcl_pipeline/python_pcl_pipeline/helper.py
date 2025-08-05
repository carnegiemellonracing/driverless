import numpy as np
import math
from dataclasses import dataclass
# Make sure you have scipy installed: pip install scipy
from scipy.spatial import KDTree
from sensor_msgs.msg import PointCloud2, PointField
import sys
import time


def point_cloud2_to_dict(msg: PointCloud2) -> dict:
    """
    Converts a sensor_msgs/PointCloud2 message to a dictionary of NumPy arrays.
    This version correctly uses the field offsets and datatypes.

    :param msg: The PointCloud2 message.
    :return: A dictionary where keys are field names ('xyz', 'rgb', 'intensity')
             and values are the corresponding NumPy arrays.
    """
    # Create a mapping from PointField datatype constants to numpy dtypes
    ros_dtype_to_np_dtype = {
        PointField.INT8: np.int8,
        PointField.UINT8: np.uint8,
        PointField.INT16: np.int16,
        PointField.UINT16: np.uint16,
        PointField.INT32: np.int32,
        PointField.UINT32: np.uint32,
        PointField.FLOAT32: np.float32,
        PointField.FLOAT64: np.float64,
    }

    # Get the raw data as a 2D array of bytes, where each row is a point.
    pc_data = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, msg.point_step)

    # Dictionary to hold the parsed fields
    parsed_fields = {}

    for field in msg.fields:
        # Get the byte offset and the numpy dtype for this field
        offset = field.offset
        dtype = ros_dtype_to_np_dtype[field.datatype]

        # The itemsize gives the number of bytes for this field's datatype
        itemsize = np.dtype(dtype).itemsize

        # Slice the data for the current field from all points
        field_data = pc_data[:, offset:offset + itemsize]

        # Interpret the bytes as the correct dtype. The .squeeze() removes
        # the trailing dimension of size 1.
        parsed_fields[field.name] = field_data.copy().view(dtype).squeeze()

    # --- Reformat the data into the desired output dictionary format ---
    output_dict = {}

    # Stack x, y, and z into a single Nx3 array
    if 'x' in parsed_fields and 'y' in parsed_fields and 'z' in parsed_fields:
        output_dict['xyz'] = np.vstack([
            parsed_fields['x'],
            parsed_fields['y'],
            parsed_fields['z']
        ]).T

    # Copy other desired fields directly
    if 'rgb' in parsed_fields:
        output_dict['rgb'] = parsed_fields['rgb']

    if 'intensity' in parsed_fields:
        output_dict['intensity'] = parsed_fields['intensity']

    return output_dict


def dict_to_point_cloud2(point_cloud_dict: dict, frame_id: str = 'base_link') -> PointCloud2:
    """
    Converts a dictionary of numpy arrays to a PointCloud2 message.
    This version uses NumPy structured arrays for correct and efficient serialization.

    :param point_cloud_dict: Dictionary containing point cloud data.
                             Must have 'xyz' (Nx3 float32).
                             Can optionally have 'rgb' (Nx1 uint32) and/or
                             'intensity' (Nx1 uint16 or float32).
    :param frame_id: The coordinate frame ID for the message header.
    :return: A sensor_msgs.msg.PointCloud2 message.
    """
    if "xyz" not in point_cloud_dict:
        raise ValueError("The input dictionary must have an 'xyz' field.")

    num_points = point_cloud_dict["xyz"].shape[0]

    # --- 1. Dynamically build the dtype and PointFields ---
    # This is the core of the new approach. We build a list of tuples for the dtype
    # and a list of PointField objects simultaneously.

    dtype_list = [('x', np.float32), ('y', np.float32), ('z', np.float32)]
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    current_offset = 12  # Start next field after x, y, z

    if "rgb" in point_cloud_dict:
        dtype_list.append(('rgb', np.uint32))
        fields.append(PointField(name='rgb', offset=current_offset, datatype=PointField.UINT32, count=1))
        current_offset += 4

    if "intensity" in point_cloud_dict:
        # Common practice is to use float32 for intensity
        intensity_dtype = np.float32
        intensity_datatype = PointField.FLOAT32
        intensity_bytes = 4

        # Or uncomment below for uint16 if you are certain
        # intensity_dtype = np.uint16
        # intensity_datatype = PointField.UINT16
        # intensity_bytes = 2

        dtype_list.append(('intensity', intensity_dtype))
        fields.append(PointField(name='intensity', offset=current_offset, datatype=intensity_datatype, count=1))
        current_offset += intensity_bytes

    # --- 2. Create the structured NumPy array ---
    structured_array = np.empty(num_points, dtype=np.dtype(dtype_list))

    # --- 3. Populate the structured array ---
    structured_array['x'] = point_cloud_dict['xyz'][:, 0]
    structured_array['y'] = point_cloud_dict['xyz'][:, 1]
    structured_array['z'] = point_cloud_dict['xyz'][:, 2]

    if "rgb" in point_cloud_dict:
        structured_array['rgb'] = point_cloud_dict['rgb']

    if "intensity" in point_cloud_dict:
        structured_array['intensity'] = point_cloud_dict['intensity']

    # --- 4. Create the PointCloud2 message ---
    msg = PointCloud2()
    msg.header.frame_id = frame_id
    current_time = time.time()
    msg.header.stamp.sec = int(current_time)
    msg.header.stamp.nanosec = int((current_time - msg.header.stamp.sec) * 1e9)

    msg.height = 1
    msg.width = num_points
    msg.is_dense = not np.isnan(point_cloud_dict["xyz"]).any()
    msg.is_bigendian = sys.byteorder != 'little'

    # Assign the dynamically created fields
    msg.fields = fields

    # Set the point_step and row_step from the structured array's itemsize
    # This is now 100% accurate and automatic.
    msg.point_step = structured_array.itemsize
    msg.row_step = msg.point_step * num_points

    # The .tobytes() method on the structured array gives the exact byte layout we need.
    msg.data = structured_array.tobytes()

    return msg


@dataclass
class Radial:
    """Represents a point in radial coordinates (angle, radius, z, intensity)."""
    angle: float
    radius: float
    z: float
    intensity: float  # <-- ADDED FIELD


def point2radial(pt: np.ndarray, intensity: float) -> Radial:  # <-- ADDED intensity ARGUMENT
    """Converts (x,y,z) and its intensity to a Radial object."""
    angle = math.atan2(pt[1], pt[0])  # pt[1] is y, pt[0] is x
    radius = math.sqrt(pt[0]**2 + pt[1]**2)
    return Radial(angle=angle, radius=radius, z=pt[2], intensity=intensity)  # <-- ADDED intensity


def radial2point(rd: Radial) -> np.ndarray:
    """Converts (radius,ang,z) to (x,y,z), ignoring intensity for geometry."""
    x = rd.radius * math.cos(rd.angle)
    y = rd.radius * math.sin(rd.angle)
    return np.array([x, y, rd.z])


def min_height(bin_points: list[Radial]) -> Radial:
    """Gets the point with the minimum z-value in a list of radial points."""
    if not bin_points:
        # Update sentinel value to include an intensity placeholder
        return Radial(angle=-100, radius=-100, z=-100, intensity=-100)  # <-- MODIFIED

    mini = bin_points[0]
    for rd in bin_points[1:]:
        if rd.z < mini.z:
            mini = rd
    return mini


def grace_and_conrad(cloud: np.ndarray, intensities: np.ndarray, alpha: float, num_bins: int, height_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Implements the GraceAndConrad ground filtering algorithm, filtering intensities as well.

    Args:
        cloud: An Nx3 numpy array of (x,y,z) points.
        intensities: An N-element numpy array of intensity values corresponding to the cloud.
        alpha: The angular size of each segment (radians).
        num_bins: The number of bins per segment.
        height_threshold: The vertical distance above the fitted ground line to keep points.

    Returns:
        A tuple containing:
        - An Mx3 numpy array of filtered ground points.
        - An M-element numpy array of corresponding filtered intensities.
    """
    # These parameters remain the same
    upper_height_threshold = 0.7
    angle_min = -1 * (math.pi/4)
    angle_max = 1 * (math.pi/4)
    radius_max = 30.
    num_segs = int((angle_max - angle_min) / alpha)

    segments = [[[] for _ in range(num_bins)] for _ in range(num_segs)]

    # 1. Convert points to radial coordinates, CARRYING intensity, and sort into bins
    for i in range(cloud.shape[0]):
        pt = cloud[i, :]
        intensity = intensities[i]  # Get the corresponding intensity
        rd = point2radial(pt, intensity)  # Pass intensity to the helper

        if rd.radius < radius_max:
            seg_index = int(rd.angle / alpha) + num_segs // 2 - (1 if rd.angle < 0 else 0)
            bin_index = int(rd.radius / (radius_max / num_bins))

            if 0 <= seg_index < num_segs and 0 <= bin_index < num_bins:
                segments[seg_index][bin_index].append(rd)

    # Create separate output lists for points and their intensities
    output_points = []
    output_intensities = []

    # 2. Process each segment (This part of the logic is unchanged)
    for seg in range(num_segs):
        minis_rad = []
        minis_z = []

        for bin_idx in range(num_bins):
            mini = min_height(segments[seg][bin_idx])
            if mini.radius != -100:
                minis_rad.append(mini.radius)
                minis_z.append(mini.z)

        # Linear regression logic remains identical
        sum_rad, sum_rad2, sum_z, sum_radz = 0.0, 0.0, 0.0, 0.0
        n = len(minis_rad)
        for i in range(n):
            rad = minis_rad[i]
            z = minis_z[i]
            sum_rad += rad
            sum_rad2 += rad * rad
            sum_z += z
            sum_radz += rad * z

        slope = 0.0
        intercept = sum_z
        if n > 1:
            denominator = (n * sum_rad2 - sum_rad * sum_rad)
            if denominator != 0:
                slope = (n * sum_radz - sum_rad * sum_z) / denominator
                intercept = (sum_z - slope * sum_rad) / n
            else:
                slope = 0
                intercept = sum_z / n

        # 3. Classify points and collect both point data and intensity data
        for bin_idx in range(num_bins):
            for pt_rd in segments[seg][bin_idx]:  # pt_rd is now a Radial object with intensity
                low_cutoff = slope * pt_rd.radius + intercept + height_threshold
                high_cutoff = slope * pt_rd.radius + intercept + upper_height_threshold

                if low_cutoff < pt_rd.z < high_cutoff:
                    # If the point is kept, keep both its geometry and its intensity
                    output_points.append(radial2point(pt_rd))
                    output_intensities.append(pt_rd.intensity)  # <-- STORE THE INTENSITY

    # Handle the case of no output points
    if not output_points:
        return np.empty((0, 3)), np.empty((0,))

    # Return a tuple of two numpy arrays
    return np.array(output_points, dtype=np.float32), np.array(output_intensities)


def expand_cluster(cloud: np.ndarray, visited: list[bool], cluster: list[int],
                   point_idx: int, neighbors: list[int], cluster_id: int,
                   epsilon: float, min_points: int):
    """Expands a cluster from a core point by exploring its neighbors."""
    cluster[point_idx] = cluster_id

    i = 0
    # Use a while loop because the neighbors list is modified during iteration
    while i < len(neighbors):
        neighbor_idx = neighbors[i]

        if not visited[neighbor_idx]:
            visited[neighbor_idx] = True
            new_neighbors = region_query(cloud, neighbor_idx, epsilon)

            if len(new_neighbors) >= min_points:
                # This neighbor is also a core point, so add its neighbors to the processing queue
                neighbors.extend(new_neighbors)

        # If neighbor is unclassified, add it to the current cluster
        if cluster[neighbor_idx] == -1:
            cluster[neighbor_idx] = cluster_id
        i += 1


def region_query(cloud: np.ndarray, point_idx: int, epsilon: float) -> list[int]:
    """Returns the indices of all points within epsilon distance of a given point."""
    neighbors = []
    point = cloud[point_idx, :]
    for i in range(cloud.shape[0]):
        if euclidean_distance(point, cloud[i, :]) <= epsilon:
            neighbors.append(i)
    return neighbors


def compute_centroids(cloud: np.ndarray, clusters: dict[int, list[int]]) -> np.ndarray:
    """Computes the geometric center (centroid) for each cluster."""
    if not clusters:
        return np.empty((0, 3))

    centroids_list = []
    for cluster_id in sorted(clusters.keys()):  # Sort for deterministic output
        indices = clusters[cluster_id]
        # Use numpy's efficient mean calculation over the specified axis
        centroid = np.mean(cloud[indices, :], axis=0)
        centroids_list.append(centroid)

    return np.array(centroids_list)


# You need the optimized expand_cluster that also uses the tree
def expand_cluster_optimized(tree, cloud, visited, cluster, point_idx, neighbors, cluster_id, epsilon, min_points):
    cluster[point_idx] = cluster_id

    i = 0
    while i < len(neighbors):
        neighbor_idx = neighbors[i]

        if not visited[neighbor_idx]:
            visited[neighbor_idx] = True

            # This neighbor search is now extremely fast
            new_neighbors = tree.query_ball_point(cloud[neighbor_idx], r=epsilon)

            if len(new_neighbors) >= min_points:
                neighbors.extend(new_neighbors)

        if cluster[neighbor_idx] == -1:
            cluster[neighbor_idx] = cluster_id
        i += 1


def dbscan_optimized(cloud: np.ndarray, epsilon: float, min_points: int) -> np.ndarray:
    """
    A robust and fast implementation of DBSCAN using a k-d tree for neighbor searches.
    This is the standard approach and will not be slow on large datasets.
    """
    num_points = cloud.shape[0]
    if num_points == 0:
        return np.empty((0, 3))

    # 1. Build the spatial index ONCE. This is very fast.
    tree = KDTree(cloud)

    visited = [False] * num_points
    cluster = [-1] * num_points
    cluster_id = 0

    for i in range(num_points):
        if visited[i]:
            continue
        visited[i] = True

        # 2. Use the tree to find neighbors. This is now O(log N) instead of O(N).
        neighbors = tree.query_ball_point(cloud[i], r=epsilon)
        # ADD THIS DIAGNOSTIC LINE

        if len(neighbors) < min_points:
            cluster[i] = 0
        else:
            cluster_id += 1
            # 3. Call the OPTIMIZED expand_cluster function
            expand_cluster_optimized(tree, cloud, visited, cluster, i, neighbors, cluster_id, epsilon, min_points)

    # ... (The rest of your code for collecting and computing centroids remains the same) ...
    clusters_map = {}
    for i, point_cluster_id in enumerate(cluster):
        if point_cluster_id > 0:
            if point_cluster_id not in clusters_map:
                clusters_map[point_cluster_id] = []
            clusters_map[point_cluster_id].append(i)

    return compute_centroids(cloud, clusters_map)


def dbscan2_optimized_and_filtered(cloud: np.ndarray, epsilon: float, min_points: int) -> np.ndarray:
    """
    Optimized DBSCAN that ACTUALLY FILTERS the cloud, returning only the points
    that belong to a cluster (i.e., removes noise).
    """
    num_points = cloud.shape[0]
    if num_points == 0:
        return np.empty((0, 3))

    tree = KDTree(cloud)
    visited = [False] * num_points
    cluster = [-1] * num_points
    cluster_id = 0

    for i in range(num_points):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = tree.query_ball_point(cloud[i], r=epsilon)
        if len(neighbors) < min_points:
            cluster[i] = 0  # Mark as noise
        else:
            cluster_id += 1
            expand_cluster_optimized(tree, cloud, visited, cluster, i, neighbors, cluster_id, epsilon, min_points)

    # --- FILTERING STEP ---
    # Create a boolean mask to select only points that are NOT noise.
    # The C++ code marks noise as cluster_id 0.
    is_not_noise = [c > 0 for c in cluster]

    # Return a new cloud containing only the points that are part of a cluster.
    return cloud[is_not_noise]


def dbscan(cloud: np.ndarray, epsilon: float, min_points: int) -> np.ndarray:
    """
    Performs DBSCAN clustering and returns the centroids of the identified clusters.
    """
    num_points = cloud.shape[0]
    if num_points == 0:
        return np.empty((0, 3))

    visited = [False] * num_points
    cluster = [-1] * num_points  # -1: unclassified, 0: noise, >0: cluster ID

    cluster_id = 0

    print(num_points)
    for i in range(num_points):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = region_query(cloud, i, epsilon)
        # ADD THIS DIAGNOSTIC LINE
        print(f"--- DIAGNOSTIC: Found {len(neighbors)} neighbors for point {i} ---")

        if len(neighbors) < min_points:
            cluster[i] = 0  # Mark as noise
        else:
            cluster_id += 1  # A new cluster is found
            expand_cluster(cloud, visited, cluster, i, neighbors, cluster_id, epsilon, min_points)

    # Collect all points belonging to each cluster ID
    clusters_map = {}
    for i, point_cluster_id in enumerate(cluster):
        if point_cluster_id > 0:  # Only collect actual clusters, not noise
            if point_cluster_id not in clusters_map:
                clusters_map[point_cluster_id] = []
            clusters_map[point_cluster_id].append(i)

    return compute_centroids(cloud, clusters_map)


def dbscan2(cloud: np.ndarray, epsilon: float, min_points: int) -> np.ndarray:
    """
    Performs DBSCAN clustering but returns the original, unmodified point cloud.
    """
    num_points = cloud.shape[0]
    if num_points == 0:
        return np.empty((0, 3))

    visited = [False] * num_points
    cluster = [-1] * num_points

    cluster_id = 0
    for i in range(num_points):
        if visited[i]:
            continue
        visited[i] = True

        neighbors = region_query(cloud, i, epsilon)

        if len(neighbors) < min_points:
            cluster[i] = 0  # Mark as noise
        else:
            cluster_id += 1
            expand_cluster(cloud, visited, cluster, i, neighbors, cluster_id, epsilon, min_points)

    # The key difference: this function returns the original cloud.
    # The clustering happens internally, but its results are not used in the return value.
    return cloud


def dbscan3_get_each_cone_cluster(cloud: np.ndarray, intensities: np.ndarray, epsilon: float, min_points: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Add a comment eventually - for Lidar cone coloring
    """
    num_points = cloud.shape[0]

    tree = KDTree(cloud)
    visited = [False] * num_points
    cluster = [-1] * num_points
    cluster_id = 0

    for i in range(num_points):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = tree.query_ball_point(cloud[i], r=epsilon)
        if len(neighbors) < min_points:
            cluster[i] = 0  # Mark as noise
        else:
            cluster_id += 1
            expand_cluster_optimized(tree, cloud, visited, cluster, i, neighbors, cluster_id, epsilon, min_points)
            
    # --- Main Data Collection Logic ---
    clusters_map = {}
    for (i, cluster_id) in enumerate(cluster):
        if cluster_id > 0:
            if cluster_id not in clusters_map:
                clusters_map[cluster_id] = {'points': [], 'intensities': []}
            
            clusters_map[cluster_id]['points'].append(cloud[i])
            clusters_map[cluster_id]['intensities'].append(intensities[i])

    list_clusters = []
    for i in sorted(clusters_map.keys()):
        list_clusters.append((np.array(clusters_map[i]['points']), np.array(clusters_map[i]['intensities'])))
        
    return list_clusters
            

def cone_cluster_to_intensity_grid(cloud: np.ndarray, intensities: np.ndarray, grid_size=(32, 32)) -> np.ndarray:
    """
    Converts single 3d cluster into 2d intensity grid for Lidar Cone Coloring
    """
    if cloud.shape[0] == 0:
        return np.zeros(grid_size, dtype=np.float32)
    
    centroid = np.mean(cloud, axis=0)
    centered_cloud = cloud - centroid
    
    # flattening algorithm:
        # take vector from lidar origin to centroid -> plane
        # project points onto plane
        
    centered_cloud = centered_cloud[:, 1:3]
    max_size = np.max(np.abs(centered_cloud))
    if max_size == 0:
        return np.zeros(grid_size, dtype=np.float32)
    
    y = ((centered_cloud[:, 0] / max_size) * (grid_size[0] / 2) + (grid_size[0] / 2)).astype(int)
    z = ((centered_cloud[:, 1] / max_size) * (grid_size[1] / 2) + (grid_size[1] / 2)).astype(int)
    
    y_shape = y.shape[0]
    z_shape = z.shape[0]
    y = np.clip(y, 0, grid_size[0] - 1)
    z = np.clip(z, 0, grid_size[1] - 1)
    print(y_shape - y.shape[0])
    print(z_shape - z.shape[0])
    
    grid = np.zeros(grid_size, dtype=np.float32)
    # grid[y,z] = intensities
    np.maximum.at(grid, (z, y), intensities)
    
    # Normalization
    # min_val, max_val = np.min(grid), np.max(grid)
    # if max_val > min_val:
    #     grid = (grid - min_val) / (max_val - min_val)

    return grid