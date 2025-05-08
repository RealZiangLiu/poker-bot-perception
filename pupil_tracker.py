import zmq
import msgpack
import time
import numpy as np
import os
from datetime import datetime
from typing import Dict, Tuple, List
from msgpack import loads, packb
import json
import pickle
import cv2
from collections import deque

class PupilTracker:
    def __init__(self, ip: str = "127.0.0.1", port: int = 50020, gaze_shift_threshold: float = 0.1, frame_rate: float = 30.0):
        """Initialize connection to Pupil Capture.
        
        Args:
            ip: IP address of Pupil Capture
            port: Port of Pupil Capture
            gaze_shift_threshold: Threshold for detecting gaze shifts
            frame_rate: Target frame rate for recording (frames per second)
        """
        self.gaze_shift_threshold = gaze_shift_threshold
        self.frame_rate = frame_rate
        self.frame_interval = 1.0 / frame_rate  # Time between frames in seconds
        self.last_frame_time = 0  # Track when we last recorded a frame
        
        self.context = zmq.Context()
        
        # Setup command socket
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{ip}:{port}")
        
        # Get subscription port
        self.socket.send_string("SUB_PORT")
        sub_port = self.socket.recv_string()
        
        # Setup subscription socket for all data
        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.connect(f"tcp://{ip}:{sub_port}")
        
        # Subscribe to all relevant topics
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "gaze")
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "fixation")
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "blink")
        self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "frame.")
        
        # Set frame format to BGR
        self.notify({"subject": "frame_publishing.set_format", "format": "bgr"})
        
        # Data collection
        self.recorded_data = []
        self.recorded_frames = []
        
        # Frame rate calculation
        self.frame_times = deque(maxlen=30)  # Store last 30 frame times
        self.current_frame = None
        self.current_gaze = None

    def draw_gaze(self, frame, gaze_pos):
        """Draw gaze position on frame."""
        if gaze_pos is None:
            return frame
        
        # Convert normalized position to pixel coordinates
        height, width = frame.shape[:2]
        x = int(gaze_pos[0] * width)
        y = int((1 - gaze_pos[1]) * height)  # Invert y coordinate
        
        # Draw crosshair
        color = (0, 255, 0)  # Green
        size = 20
        thickness = 2
        cv2.line(frame, (x - size, y), (x + size, y), color, thickness)
        cv2.line(frame, (x, y - size), (x, y + size), color, thickness)
        
        # Draw circle
        cv2.circle(frame, (x, y), 5, color, -1)
        
        return frame

    def calculate_fps(self):
        """Calculate current frame rate."""
        if len(self.frame_times) < 2:
            return 0
        return len(self.frame_times) / (self.frame_times[-1] - self.frame_times[0])

    def notify(self, notification):
        """Sends notification to Pupil Remote"""
        topic = "notify." + notification["subject"]
        payload = packb(notification, use_bin_type=True)
        self.socket.send_string(topic, flags=zmq.SNDMORE)
        self.socket.send(payload)
        return self.socket.recv_string()

    def recv_from_sub(self):
        """Recv a message with topic, payload."""
        topic = self.sub_socket.recv_string()
        payload = loads(self.sub_socket.recv(), raw=False)
        extra_frames = []
        while self.sub_socket.get(zmq.RCVMORE):
            extra_frames.append(self.sub_socket.recv())
        if extra_frames:
            payload["__raw_data__"] = extra_frames
        return topic, payload

    def has_new_data_available(self):
        """Returns True as long subscription socket has received data queued for processing"""
        return self.sub_socket.get(zmq.EVENTS) & zmq.POLLIN

    def process_data(self):
        """Process recorded data and calculate metrics."""
        # Group fixations by ID
        fixation_groups = {}
        blinks = []
        gaze_data = []
        
        for record in self.recorded_data:
            if record['topic'] == "fixations":
                fixation_id = record['data'].get('id')
                if fixation_id is not None:
                    if fixation_id not in fixation_groups:
                        fixation_groups[fixation_id] = {
                            'start_time': record['data'].get('timestamp'),
                            'end_time': record['data'].get('timestamp'),
                            'data': record['data']
                        }
                    else:
                        # Update end time if this timestamp is later
                        current_time = record['data'].get('timestamp')
                        if current_time > fixation_groups[fixation_id]['end_time']:
                            fixation_groups[fixation_id]['end_time'] = current_time
            elif record['topic'] == "blinks" and record['data'].get('type') == 'onset' and record['data'].get('confidence') > 0.9:
                blinks.append(record['data'])
            elif record['topic'] == "gaze.3d.0.":
                gaze_data.append(record['data'])
        
        # Convert grouped fixations to list and calculate durations
        fixations = []
        for fixation in fixation_groups.values():
            duration = fixation['end_time'] - fixation['start_time']
            fixations.append({
                'duration': duration,
                'data': fixation['data']
            })
        
        metrics = {}
        
        # Fixation metrics
        if fixations:
            fixation_durations = [f['duration'] for f in fixations]  # Already in seconds
            metrics['mean_fixation_duration'] = np.mean(fixation_durations)
            metrics['fixation_count'] = len(fixations)
            metrics['total_fixation_time'] = sum(fixation_durations)
        
        # Blink metrics
        metrics['blink_count'] = len(blinks)
        
        # Gaze shift metrics
        gaze_shifts = 0
        if len(gaze_data) > 1:
            gaze_positions = [(g.get('norm_pos', (0, 0))[0], g.get('norm_pos', (0, 0))[1]) 
                            for g in gaze_data]
            for i in range(1, len(gaze_positions)):
                dx = abs(gaze_positions[i][0] - gaze_positions[i-1][0])
                dy = abs(gaze_positions[i][1] - gaze_positions[i-1][1])
                if dx > self.gaze_shift_threshold or dy > self.gaze_shift_threshold:
                    gaze_shifts += 1
        
        if gaze_data:
            duration = gaze_data[-1]['timestamp'] - gaze_data[0]['timestamp']
            metrics['gaze_shifts_per_second'] = gaze_shifts / duration if duration > 0 else 0
            metrics['blinks_per_second'] = len(blinks) / duration if duration > 0 else 0
        
        return metrics

    def clean_data_for_json(self, data):
        """Clean data to make it JSON serializable."""
        if isinstance(data, dict):
            return {k: self.clean_data_for_json(v) for k, v in data.items() if k != '__raw_data__'}
        elif isinstance(data, list):
            return [self.clean_data_for_json(item) for item in data]
        elif isinstance(data, (str, int, float, bool, type(None))):
            return data
        elif isinstance(data, bytes):
            return "<binary data>"
        elif isinstance(data, np.ndarray):
            return data.tolist()
        else:
            return str(data)

    def save_data(self):
        """Save recording data and metrics to files."""
        try:
            # Create recordings directory if it doesn't exist
            recordings_dir = os.path.abspath('./recordings')
            os.makedirs(recordings_dir, mode=0o777, exist_ok=True)
            
            # Generate timestamp for filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            session_dir = os.path.join(recordings_dir, f'session_{timestamp}')
            os.makedirs(session_dir, mode=0o777, exist_ok=True)
            
            # Process and save metrics
            metrics = self.process_data()
            metrics_file = os.path.join(session_dir, 'metrics.json')
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Clean and save raw data
            cleaned_data = [{
                'timestamp': record['timestamp'],
                'topic': record['topic'],
                'data': self.clean_data_for_json(record['data'])
            } for record in self.recorded_data]
            
            data_file = os.path.join(session_dir, 'raw_data.json')
            with open(data_file, 'w') as f:
                json.dump(cleaned_data, f, indent=2)
            
            # Save frames with timestamps
            frames_file = os.path.join(session_dir, 'frames.pkl')
            with open(frames_file, 'wb') as f:
                pickle.dump(self.recorded_frames, f)
            
            print(f"\nData saved to session directory: {session_dir}")
            print(f"Metrics: {metrics_file}")
            print(f"Raw data: {data_file}")
            print(f"Frames: {frames_file}")
        except PermissionError as e:
            print(f"Error: Permission denied when creating directories. {e}")
            print("Please ensure you have write permissions in the current directory.")
            raise

    def close(self):
        """Close all connections."""
        self.socket.close()
        self.sub_socket.close()
        self.context.term()
        cv2.destroyAllWindows()

def main():
    tracker = PupilTracker(frame_rate=10.0)  # Example: record at 10 FPS
    print("\nStarting data collection...")
    print("Press Ctrl+C to stop and save data")
    
    try:
        while True:
            while tracker.has_new_data_available():
                topic, msg = tracker.recv_from_sub()
                
                if topic.startswith("frame.") and msg["format"] != "bgr":
                    continue
                
                if topic == "frame.world":
                    # Only record frame if enough time has passed
                    if msg['timestamp'] - tracker.last_frame_time >= tracker.frame_interval:
                        frame = np.frombuffer(msg["__raw_data__"][0], dtype=np.uint8).reshape(msg["height"], msg["width"], 3)
                        tracker.current_frame = frame
                        tracker.frame_times.append(msg['timestamp'])
                        tracker.recorded_frames.append({
                            'timestamp': msg['timestamp'],
                            'frame': frame
                        })
                        tracker.last_frame_time = msg['timestamp']
                elif topic == "gaze.3d.0.":
                    tracker.current_gaze = msg.get('norm_pos')
                
                tracker.recorded_data.append({
                    'timestamp': msg['timestamp'],
                    'topic': topic,
                    'data': msg
                })
            
            # Display frame with gaze overlay
            if tracker.current_frame is not None:
                # Draw gaze position
                frame = tracker.draw_gaze(tracker.current_frame.copy(), tracker.current_gaze)
                
                # Add FPS text
                fps = tracker.calculate_fps()
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow("Eye Tracking", frame)
                cv2.waitKey(1)
                
    except KeyboardInterrupt:
        print("\nStopping data collection...")
        tracker.save_data()
    finally:
        tracker.close()

if __name__ == "__main__":
    main() 