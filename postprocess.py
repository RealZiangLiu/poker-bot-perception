import os
import json
import pickle
import numpy as np
import cv2
from pupil_apriltags import Detector
import glob

# Configuration for tag mapping and thresholds
TAG_MAPPING = {
    "robot": 0,
    "cards": 3
}

# Separate distance thresholds for each object
DISTANCE_THRESHOLDS = {
    "robot": 0.5,  # 15% of screen size
    "cards": 0.5  # 25% of screen size
}

class PostProcessor:
    def __init__(self, session_dir):
        """Initialize post-processor with session directory."""
        self.session_dir = session_dir
        
        # Load recorded data
        self.load_recorded_data()
        
        # Initialize AprilTag detector
        self.detector = Detector(
            families="tag36h11",
            nthreads=4,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=True,
            decode_sharpening=0.25,
            debug=0
        )
        
        # Store results
        self.tag_positions = {}  # timestamp -> list of (tag_id, center_x, center_y)
        self.gaze_distances = {}  # tag_id -> list of (timestamp, distance)
        self.looking_at = {}  # timestamp -> "robot", "cards", or None

    def load_recorded_data(self):
        """Load recorded data and frames from session directory."""
        # Load raw data
        with open(os.path.join(self.session_dir, 'raw_data.json'), 'r') as f:
            self.raw_data = json.load(f)
        
        # Load frames
        with open(os.path.join(self.session_dir, 'frames.pkl'), 'rb') as f:
            self.frames = pickle.load(f)
        
        # Organize data by timestamp
        self.gaze_data = []
        self.frame_data = {}
        
        # Load frames
        for frame in self.frames:
            self.frame_data[frame['timestamp']] = frame['frame']
        
        # Load gaze data
        for record in self.raw_data:
            if record['topic'] == "gaze.3d.0.":
                self.gaze_data.append({
                    'timestamp': record['timestamp'],
                    'data': record['data']
                })
        
        # Sort data by timestamp
        self.gaze_data.sort(key=lambda x: x['timestamp'])
        self.frame_timestamps = sorted(self.frame_data.keys())

    def detect_tags(self, frame):
        """Detect AprilTags in a frame and return their centers."""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect tags
        results = self.detector.detect(gray)
        
        # Extract tag centers
        tag_centers = []
        for tag in results:
            # Get center point of tag
            center_x = np.mean(tag.corners[:, 0])
            center_y = np.mean(tag.corners[:, 1])
            tag_centers.append((tag.tag_id, center_x, center_y))
        
        return tag_centers

    def process_frame(self, frame, timestamp):
        """Process a single frame and detect tag positions."""
        # Detect tags
        tag_centers = self.detect_tags(frame)
        
        # Store tag positions
        self.tag_positions[timestamp] = tag_centers

    def process_gaze(self, gaze_data, frame_timestamp):
        """Process gaze data relative to detected tags in the nearest frame."""
        # Get tag positions for this frame
        frame_tags = self.tag_positions.get(frame_timestamp, [])
        if not frame_tags:
            return
        
        # Get gaze position in pixel coordinates
        gaze_x = gaze_data['data'].get('norm_pos', [0.5, 0.5])[0] * self.frame_data[frame_timestamp].shape[1]
        gaze_y = gaze_data['data'].get('norm_pos', [0.5, 0.5])[1] * self.frame_data[frame_timestamp].shape[0]
        
        # Calculate distance to each tag
        min_distance = float('inf')
        closest_tag = None
        closest_tag_distance = None
        
        for tag_id, tag_x, tag_y in frame_tags:
            # Calculate normalized distance (0 to 1)
            dx = (gaze_x - tag_x) / self.frame_data[frame_timestamp].shape[1]
            dy = (gaze_y - tag_y) / self.frame_data[frame_timestamp].shape[0]
            distance = np.sqrt(dx*dx + dy*dy)
            
            # Store result
            if tag_id not in self.gaze_distances:
                self.gaze_distances[tag_id] = []
            
            self.gaze_distances[tag_id].append({
                'timestamp': gaze_data['timestamp'],
                'frame_timestamp': frame_timestamp,
                'gaze_pos': [gaze_x, gaze_y],
                'tag_pos': [tag_x, tag_y],
                'distance': float(distance)  # Convert to float for JSON serialization
            })
            
            # Track closest tag
            if distance < min_distance:
                min_distance = distance
                closest_tag = tag_id
                closest_tag_distance = distance
        
        # Determine what the user is looking at based on closest tag
        if closest_tag is not None:
            # Find the object name for this tag
            for obj_name, tag_id in TAG_MAPPING.items():
                if tag_id == closest_tag:
                    # Check against this object's specific threshold
                    if closest_tag_distance <= DISTANCE_THRESHOLDS[obj_name]:
                        self.looking_at[gaze_data['timestamp']] = obj_name
                    else:
                        self.looking_at[gaze_data['timestamp']] = None
                    break
        else:
            self.looking_at[gaze_data['timestamp']] = None

    def process_all(self):
        """Process all frames and gaze data."""
        print("Processing frames...")
        for timestamp, frame in self.frame_data.items():
            self.process_frame(frame, timestamp)
        
        print("Processing gaze data...")
        # Maximum allowed time difference between gaze and frame (in seconds)
        MAX_TIME_DIFF = 0.1  # 100ms
        
        for gaze in self.gaze_data:
            # Find the nearest frame timestamp for this gaze point
            nearest_frame = min(self.frame_timestamps, 
                              key=lambda x: abs(x - gaze['timestamp']))
            
            # Check if the timestamps are close enough
            time_diff = abs(nearest_frame - gaze['timestamp'])
            if time_diff <= MAX_TIME_DIFF:
                self.process_gaze(gaze, nearest_frame)
            else:
                print(f"Skipping gaze point - too far from nearest frame: {time_diff:.3f}s")
                self.looking_at[gaze['timestamp']] = None
        
        # Calculate percentages
        total_samples = len(self.looking_at)
        robot_count = sum(1 for obj in self.looking_at.values() if obj == "robot")
        cards_count = sum(1 for obj in self.looking_at.values() if obj == "cards")
        
        robot_percentage = (robot_count / total_samples * 100) if total_samples > 0 else 0.0
        cards_percentage = (cards_count / total_samples * 100) if total_samples > 0 else 0.0
        
        print("\nLooking Statistics:")
        print(f"Total samples: {total_samples}")
        print(f"Looking at robot: {robot_percentage:.1f}% (threshold: {DISTANCE_THRESHOLDS['robot']*100:.0f}% of screen)")
        print(f"Looking at cards: {cards_percentage:.1f}% (threshold: {DISTANCE_THRESHOLDS['cards']*100:.0f}% of screen)")
        print(f"Looking elsewhere: {100 - robot_percentage - cards_percentage:.1f}%")
        
        print("Processing complete!")

    def save_results(self):
        """Save processing results to JSON file."""
        # Convert data to JSON serializable format
        json_data = {
            'tag_positions': self.tag_positions,
            'gaze_distances': self.gaze_distances,
            'looking_at': self.looking_at,
            'config': {
                'tag_mapping': TAG_MAPPING,
                'distance_thresholds': DISTANCE_THRESHOLDS
            }
        }
        
        # Save to JSON file
        output_file = os.path.join(self.session_dir, 'processing_results.json')
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Results saved to {output_file}")
        
        # Load metrics from metrics.json
        metrics_file = os.path.join(self.session_dir, 'metrics.json')
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        # Calculate gaze percentages
        total_samples = len(self.looking_at)
        robot_count = sum(1 for obj in self.looking_at.values() if obj == "robot")
        cards_count = sum(1 for obj in self.looking_at.values() if obj == "cards")
        
        robot_percentage = (robot_count / total_samples * 100) if total_samples > 0 else 0.0
        cards_percentage = (cards_count / total_samples * 100) if total_samples > 0 else 0.0
        
        # Create nonverbal behavior data
        nonverbal_data = {
            "gaze_cards_percentage": round(cards_percentage, 1),
            "gaze_robot_percentage": round(robot_percentage, 1),
            "gaze_shifts_per_second": round(metrics.get('gaze_shifts_per_second', 0.0), 1),
            "gaze_mean_fixation_duration": round(metrics.get('mean_fixation_duration', 0.0), 1),
            "head_pose_shifts_rate_per_second": 0.0,  # Placeholder value
            "blinks_per_second": round(metrics.get('blinks_per_second', 0.0), 1)
        }
        
        # Save nonverbal behavior data to session directory
        nonverbal_file = os.path.join(self.session_dir, 'nonverbal_behavior.json')
        with open(nonverbal_file, 'w') as f:
            json.dump(nonverbal_data, f, indent=2)
        
        # Create data directory if it doesn't exist
        data_dir = os.path.abspath('./data')
        os.makedirs(data_dir, exist_ok=True)
        
        # Save copy to data directory
        data_file = os.path.join(data_dir, 'nonverbal_behavior.json')
        with open(data_file, 'w') as f:
            json.dump(nonverbal_data, f, indent=2)
        
        print(f"Nonverbal behavior data saved to {nonverbal_file}")
        print(f"Copy saved to {data_file}")

def main():
    # Find the most recent session directory
    recordings_dir = os.path.abspath('./recordings')
    session_dirs = glob.glob(os.path.join(recordings_dir, 'session_*'))
    if not session_dirs:
        print("No session directories found!")
        return
    
    latest_session = max(session_dirs, key=os.path.getctime)
    print(f"Processing session: {latest_session}")
    
    # Create and run post-processor
    processor = PostProcessor(latest_session)
    processor.process_all()
    processor.save_results()

if __name__ == "__main__":
    main() 