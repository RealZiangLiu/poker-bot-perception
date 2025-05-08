import os
from pathlib import Path
import msgpack
import numpy as np
import gc

def load_intrinsics(file_path="~/pupil_capture_settings/Pupil_Cam1_ID2.intrinsics"):
    """Load camera intrinsics from Pupil Capture settings file."""
    file_path = Path(file_path).expanduser()
    
    try:
        with file_path.open("rb") as fh:
            gc.disable()  # speeds deserialization up
            data = msgpack.unpack(fh)
            gc.enable()
            
            # Get the resolution key (it's a tuple in bytes format)
            resolution_key = next(k for k in data.keys() if k != b'version')
            
            # Extract camera parameters from the nested structure
            camera_data = data[resolution_key]
            camera_matrix = np.array(camera_data[b'camera_matrix'])
            dist_coeffs = np.array(camera_data[b'dist_coefs'])
            resolution = tuple(camera_data[b'resolution'])
            cam_type = camera_data[b'cam_type'].decode('utf-8')
            
            return {
                'camera_matrix': camera_matrix,
                'dist_coeffs': dist_coeffs,
                'resolution': resolution,
                'cam_type': cam_type
            }
    except Exception as e:
        print(f"Error loading intrinsics: {e}")
        return None

def main():
    # Load intrinsics
    intrinsics = load_intrinsics()
    
    if intrinsics is None:
        print("Failed to load camera intrinsics")
        return
    
    # Extract parameters from camera matrix
    fx = intrinsics['camera_matrix'][0,0]
    fy = intrinsics['camera_matrix'][1,1]
    cx = intrinsics['camera_matrix'][0,2]
    cy = intrinsics['camera_matrix'][1,2]
    
    # Print in a clean format
    print(f"fx: {fx:.2f}")
    print(f"fy: {fy:.2f}")
    print(f"cx: {cx:.2f}")
    print(f"cy: {cy:.2f}")

if __name__ == "__main__":
    main() 