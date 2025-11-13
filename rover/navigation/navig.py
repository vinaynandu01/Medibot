# -*- coding: utf-8 -*-
import cv2
import torch
import numpy as np
import time
import os
import pickle
import json
import math
import requests
from typing import Tuple, List, Optional
from collections import defaultdict, deque

try:
    DEPTH_SCALE = np.load("depth_scale_factor.npy").item()
except Exception:
    DEPTH_SCALE = 5.0

# ═══════════════════════════════════════════════════════════════════════════
# MIDAS DEPTH MODEL LOADER
# ═══════════════════════════════════════════════════════════════════════════

def load_midas_model():
    """Load MiDaS depth estimation model"""
    print("🔄 Loading MiDaS depth model...")
    print("🔧 Initializing MiDaS...")
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.small_transform
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device).eval()
    return midas, transform, device


# ═══════════════════════════════════════════════════════════════════════════
# ROVER CONTROL (Updated with cmds.py integration)
# ═══════════════════════════════════════════════════════════════════════════

class RoverController:
    """Controls the actual rover hardware via HTTP commands (integrated with cmds.py)"""
    
    def __init__(self, rover_ip="10.226.57.127", rover_port=8080, enable_control=True):
        self.rover_ip = rover_ip
        self.rover_port = rover_port
        self.enable_control = enable_control  # Safety flag to disable actual commands
        self.last_command_time = 0  # When was last command sent (initialize to 0)
        self.command_cooldown = 0.3  # Minimum time between commands (reduced for responsiveness)
        self.emergency_stop = False  # Emergency stop flag
        self.last_command_sent = None  # Track what command was last sent
        self.is_rotating = False  # Flag to indicate rotation in progress
        self.rotation_start_time = 0  # When rotation started
        
    def send_command(self, command):
        """
        Send command to rover (matches cmds.py behavior)
        command: 'forward', 'backward', 'left', 'right', or 'stop'
        
        Note: 
        - 'forward' moves 0.3m over 2 seconds
        - 'left'/'right' rotates 90 degrees
        - Each command is self-contained
        """
        if not self.enable_control:
            print(f"🔒 [SIMULATION] Command: {command}")
            return True
        
        # Check emergency stop
        if self.emergency_stop and command != 'stop':
            print(f"🚨 Emergency stop active - ignoring command: {command}")
            return False
            
        try:
            response = requests.post(
                f"http://{self.rover_ip}:{self.rover_port}/move",
                data={"direction": command},
                timeout=3
            )
            print(f"✅ Sent to rover: {command} | Response: {response.json()}")
            return True
        except Exception as e:
            print(f"❌ Rover control error: {e}")
            return False
    
    def execute_navigation_command(self, nav_direction):
        """
        Execute navigation command based on direction
        Handles rotation with proper stop -> rotate -> delay -> resume sequence
        """
        current_time = time.time()
        command = nav_direction.get('command', 'unknown')
        
        # Check emergency stop
        if self.emergency_stop:
            print("🚨 EMERGENCY STOP ACTIVE - No commands executed")
            return False
        
        # Check if we're currently in a rotation sequence
        if self.is_rotating:
            # Wait for rotation to complete (2 seconds after rotation command)
            if current_time - self.rotation_start_time < 2.0:
                return False  # Still in rotation cooldown
            else:
                # Rotation complete, reset flag
                self.is_rotating = False
                print("✅ Rotation sequence complete - resuming navigation")
        
        # Execute the command
        if command == 'forward':
            print(f"🚀 Forward → Moving 0.3m (2 sec)")
            success = self.send_command('forward')
            if success:
                self.last_command_sent = 'forward'
                self.last_command_time = current_time
            return success
            
        elif command in ['left', 'right']:
            # ROTATION SEQUENCE: Stop -> Rotate -> 2sec Delay
            print(f"\n{'='*60}")
            print(f"🔄 ROTATION SEQUENCE INITIATED: {command.upper()}")
            print(f"{'='*60}")
            
            # Step 1: Send STOP command first
            print(f"🛑 Step 1: Stopping rover before rotation...")
            self.send_command('stop')
            time.sleep(0.5)  # Brief pause to ensure stop is executed
            
            # Step 2: Execute rotation
            print(f"� Step 2: Rotating {command} → 90°")
            success = self.send_command(command)
            
            if success:
                self.last_command_sent = command
                self.last_command_time = current_time
                self.is_rotating = True
                self.rotation_start_time = current_time
                
                # Step 3: 2-second delay (pause navigation)
                print(f"⏳ Step 3: 2-second delay (navigation paused)...")
                time.sleep(2.0)  # ACTUAL 2-second delay
                
                print(f"{'='*60}")
                print(f"✅ Rotation complete - navigation will resume")
                print(f"{'='*60}\n")
                
                # Reset rotation flag after delay
                self.is_rotating = False
            
            return success
            
        elif command == 'stop':
            print("🛑 Stop command")
            success = self.send_command('stop')
            if success:
                self.last_command_sent = 'stop'
                self.last_command_time = current_time
            return success
        
        return False
    
    def emergency_stop_now(self):
        """Activate emergency stop immediately"""
        self.emergency_stop = True
        print("\n" + "="*80)
        print("🚨 EMERGENCY STOP ACTIVATED!")
        print("="*80)
        self.send_command('stop')
        return True
    
    def reset_emergency_stop(self):
        """Reset emergency stop flag"""
        self.emergency_stop = False
        print("✅ Emergency stop cleared - ready to resume")


# ═══════════════════════════════════════════════════════════════════════════
# KEYFRAME & STORAGE
# ═══════════════════════════════════════════════════════════════════════════

class KeyFrame:
    def __init__(self, frame_id, pose, features_3d, descriptors,
                 keypoints_2d, fx, fy, cx, cy, depth_scale=DEPTH_SCALE, max_depth=3.0):
        self.id = frame_id
        self.pose = pose.copy()
        self.features_3d = features_3d
        self.descriptors = descriptors
        self.keypoints_2d = keypoints_2d
        self.timestamp = time.time()
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy
        self.depth_scale = depth_scale
        self.max_depth = max_depth

    @staticmethod
    def from_dict(data):
        descriptors = None
        if data.get('descriptors') is not None:
            descriptors = np.array(data['descriptors'], dtype=np.uint8)
        kf = KeyFrame(
            frame_id=data['id'],
            pose=np.array(data['pose']),
            features_3d=data.get('features_3d', []),
            descriptors=descriptors,
            keypoints_2d=data.get('keypoints_2d', []),
            fx=data.get('fx', 600.0),
            fy=data.get('fy', 600.0),
            cx=data.get('cx', 320.0),
            cy=data.get('cy', 240.0),
            depth_scale=DEPTH_SCALE,
            max_depth=3.0
        )
        kf.timestamp = data.get('timestamp', time.time())
        return kf


class KeyframeStorageManager:
    def __init__(self, storage_dir="keyframes_storage"):
        self.storage_dir = storage_dir
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)
        self.index_file = os.path.join(self.storage_dir, "keyframes_index.json")
        self.index = self._load_index()

    def _load_index(self):
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {'keyframes': {}}
        return {'keyframes': {}}

    def get_all_keyframes(self):
        return sorted([int(kf_id) for kf_id in self.index.get('keyframes', {}).keys()])

    def load_keyframe(self, kf_id):
        try:
            if str(kf_id) not in self.index.get('keyframes', {}):
                return None
            filename = self.index['keyframes'][str(kf_id)]['filename']
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            return KeyFrame.from_dict(data)
        except Exception:
            return None


# ═══════════════════════════════════════════════════════════════════════════
# LIVE NAVIGATION SYSTEM WITH LOCALIZATION
# ═══════════════════════════════════════════════════════════════════════════

class LiveNavigationSystem:
    def __init__(self, fx=600.0, fy=600.0, cx=320.0, cy=240.0, 
                 enable_rover_control=False, rover_ip="10.226.57.127", rover_port=8080,
                 use_rover_stream=False, destination_kf_id=None):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
        self.storage_manager = KeyframeStorageManager()
        
        # Load MiDaS depth model
        self.midas, self.midas_transform, self.device = load_midas_model()
        
        # ORB for feature extraction
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.orb = cv2.ORB_create(nfeatures=500)
        
        # Map data
        self.all_keyframes = {}
        self.spatial_grid = {}
        self.keyframe_directions = {}  # Load from direction_map.json
        
        # Current state
        self.current_pose = None
        self.current_kf_id = None
        self.trajectory = []
        self.total_distance = 0.0
        
        # Navigation target
        self.destination_kf_id = destination_kf_id  # Target destination keyframe
        self.start_kf_id = None  # Will be set from first localization
        self.destination_reached = False
        
        # Search behavior when lost (no localization match)
        self.lost_frame_count = 0  # Counter for consecutive lost frames
        self.search_forward_limit = 3  # Try moving forward 3 times before turning
        
        # Rover control integration
        self.rover_controller = RoverController(
            rover_ip=rover_ip, 
            rover_port=rover_port, 
            enable_control=enable_rover_control
        )
        self.auto_control = enable_rover_control  # Flag to enable automatic control
        
        # Rover stream settings
        self.use_rover_stream = use_rover_stream
        self.rover_stream_url = f"http://{rover_ip}:{rover_port}/video_feed" if use_rover_stream else None
        
        # Visualization
        self.map_canvas_size = 900
        self.map_scale = 100

    def estimate_depth(self, frame):
        """Estimate depth map using MiDaS"""
        input_batch = self.midas_transform(frame).to(self.device)
        
        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        return depth_map

    def feature_extraction(self, gray):
        """Extract ORB features from grayscale image"""
        enhanced = self.clahe.apply(gray)
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=5)
        kp, desc = self.orb.detectAndCompute(enhanced, None)
        if kp is None or desc is None or len(kp) == 0:
            return [], None
        
        # Keep best by response
        pairs = list(zip(kp, desc))
        pairs.sort(key=lambda x: -x[0].response)
        top = pairs[:min(len(pairs), 500)]
        kp = [p[0] for p in top]
        desc = np.array([p[1] for p in top], dtype=np.uint8)
        return kp, desc

    def load_map(self, storage_dir="keyframes_storage"):
        """Load all keyframes from storage"""
        print(f"📂 Loading map from {storage_dir}...")
        all_kf_ids = self.storage_manager.get_all_keyframes()
        
        if not all_kf_ids:
            print("❌ No keyframes found")
            return False
        
        for kf_id in all_kf_ids:
            kf = self.storage_manager.load_keyframe(kf_id)
            if kf:
                self.all_keyframes[kf_id] = kf
                
                # Add features to spatial grid
                for x_world, y_world, z_depth in kf.features_3d:
                    key = self._spatial_grid_key(x_world, y_world)
                    if key not in self.spatial_grid:
                        self.spatial_grid[key] = {
                            'x': x_world, 'y': y_world, 'z': z_depth
                        }
        
        print(f"✅ Loaded {len(self.all_keyframes)} keyframes")
        
        # Load direction map
        self._load_direction_map(storage_dir)
        
        return True
    
    def _load_direction_map(self, storage_dir):
        """Load navigation direction map from JSON file"""
        try:
            direction_map_file = os.path.join(storage_dir, "direction_map.json")
            if os.path.exists(direction_map_file):
                with open(direction_map_file, 'r') as f:
                    data = json.load(f)
                    # Convert string keys to integers
                    self.keyframe_directions = {int(k): v for k, v in data.get('keyframe_directions', {}).items()}
                print(f"✅ Loaded direction map with {len(self.keyframe_directions)} entries")
            else:
                print("⚠️ No direction map found - will calculate directions on-the-fly")
        except Exception as e:
            print(f"⚠️ Could not load direction map: {e}")

    def _spatial_grid_key(self, x, y):
        return (int(x / 0.10), int(y / 0.10))

    def localize_frame(self, frame):
        """
        Localize current frame against all keyframes using MiDaS depth + matching.
        Returns: (matched_kf_id, confidence, inliers, estimated_pose)
        """
        # Estimate depth for current frame using MiDaS
        depth_map = self.estimate_depth(frame)
        
        # Extract features from current frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_test, desc_test = self.feature_extraction(gray)
        
        if desc_test is None or len(kp_test) == 0:
            return None, 0.0, 0, None

        # Generate 3D features from current frame using MiDaS depth
        features_3d_current = []
        for kp in kp_test:
            u, v = int(kp.pt[0]), int(kp.pt[1])
            if 0 <= v < depth_map.shape[0] and 0 <= u < depth_map.shape[1]:
                depth_val = depth_map[v, u]
                # Convert depth to metric using DEPTH_SCALE
                z = depth_val / DEPTH_SCALE
                if z > 0.1 and z < 10.0:  # Valid depth range
                    x = (u - self.cx) * z / self.fx
                    y = (v - self.cy) * z / self.fy
                    features_3d_current.append([x, y, z])
                else:
                    features_3d_current.append([0, 0, 0])  # Invalid depth
            else:
                features_3d_current.append([0, 0, 0])

        best_match_kf_id = None
        best_confidence = 0.0
        best_inliers = 0
        best_pose = None

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        for kf_id, kf in self.all_keyframes.items():
            if kf.descriptors is None:
                continue

            try:
                desc_test_uint8 = np.array(desc_test, dtype=np.uint8)
                kf_descriptors = np.array(kf.descriptors, dtype=np.uint8)
                matches = bf.knnMatch(desc_test_uint8, kf_descriptors, k=2)
            except cv2.error:
                continue

            # Lowe's ratio test
            good_matches = []
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            if len(good_matches) < 5:
                continue

            match_confidence = len(good_matches) / (len(kf.descriptors) + 1e-6)

            # Build 3D–2D correspondences using keyframe's stored 3D features
            object_points, image_points = [], []
            for m in good_matches:
                train_idx = m.trainIdx
                query_idx = m.queryIdx
                if train_idx < len(kf.features_3d):
                    x, y, z = kf.features_3d[train_idx]
                    object_points.append([x, y, z])
                    if query_idx < len(kp_test):
                        u, v = kp_test[query_idx].pt
                        image_points.append([u, v])

            if len(object_points) < 5:
                continue

            object_points = np.array(object_points, dtype=np.float32)
            image_points = np.array(image_points, dtype=np.float32)

            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                object_points, image_points, self.K, None,
                iterationsCount=100,
                reprojectionError=8.0,
                confidence=0.99
            )

            inlier_count = len(inliers) if (success and inliers is not None) else 0

            # Priority: confidence first, inliers as tie-breaker
            if match_confidence < 0.10:
                continue
            
            if (match_confidence > best_confidence) or (
                abs(match_confidence - best_confidence) < 1e-4 and inlier_count > best_inliers
            ):
                best_confidence = match_confidence
                best_inliers = inlier_count
                best_match_kf_id = kf_id
                best_pose = kf.pose.copy()

        return best_match_kf_id, best_confidence, best_inliers, best_pose

    def _snap_angle(self, angle):
        """Snap angle to nearest standard value (45, 90, 180)"""
        if angle < 22.5:
            return 0
        elif angle < 67.5:
            return 45
        elif angle < 135:
            return 90
        else:
            return 180

    def get_navigation_direction(self):
        """Get direction to move from current position towards destination keyframe"""
        if self.current_pose is None or len(self.all_keyframes) == 0:
            return {'direction': 'stop', 'angle': 0, 'distance': 0, 'target_frame': None, 
                    'status': 'no_pose', 'command': 'stop'}
        
        # Check if destination reached
        if self.destination_kf_id is not None and self.current_kf_id == self.destination_kf_id:
            if not self.destination_reached:
                self.destination_reached = True
                print(f"\n{'='*80}")
                print(f"🎯 DESTINATION REACHED! Arrived at Keyframe #{self.destination_kf_id}")
                print(f"{'='*80}\n")
            return {'direction': 'stop', 'angle': 0, 'distance': 0, 
                    'target_frame': self.destination_kf_id, 
                    'status': 'destination_reached', 'command': 'stop'}
        
        current_x, current_y, current_yaw = self.current_pose
        current_position = np.array([current_x, current_y])
        
        # Find current keyframe (nearest)
        min_dist = float('inf')
        current_kf_id = None
        
        for kf_id, kf in self.all_keyframes.items():
            kf_x, kf_y, _ = kf.pose
            dist = np.linalg.norm(np.array([kf_x, kf_y]) - current_position)
            if dist < min_dist:
                min_dist = dist
                current_kf_id = kf_id
        
        if current_kf_id is None:
            return {'direction': 'stop', 'angle': 0, 'distance': 0, 'target_frame': None, 
                    'status': 'no_keyframe', 'command': 'stop'}
        
        # Set start keyframe on first localization
        if self.start_kf_id is None:
            self.start_kf_id = current_kf_id
            print(f"\n📍 Start position: Keyframe #{self.start_kf_id}")
            print(f"🎯 Destination: Keyframe #{self.destination_kf_id}\n")
        
        # If very close to current keyframe, use pre-calculated direction if available
        if min_dist < 0.5:  # Within 0.5m of keyframe
            # Try to use pre-calculated direction from map building
            if current_kf_id in self.keyframe_directions:
                stored_direction = self.keyframe_directions[current_kf_id]
                
                # Check if this is towards destination
                next_kf = stored_direction.get('target_frame', current_kf_id + 1)
                
                # If we're at or past destination, stop
                if self.destination_kf_id is not None:
                    if current_kf_id >= self.destination_kf_id:
                        return {'direction': 'stop', 'angle': 0, 'distance': 0, 
                                'target_frame': self.destination_kf_id,
                                'status': 'destination_reached', 'command': 'stop'}
                    
                    # Don't go past destination
                    if next_kf > self.destination_kf_id:
                        return {'direction': 'stop', 'angle': 0, 'distance': 0, 
                                'target_frame': self.destination_kf_id,
                                'status': 'destination_reached', 'command': 'stop'}
                
                # Convert stored direction to proper command
                direction = stored_direction['direction']
                if direction in ['rotate_left', 'forward_left']:
                    command = 'left'
                elif direction in ['rotate_right', 'forward_right']:
                    command = 'right'
                else:
                    command = direction  # 'forward' or 'stop'
                
                return {
                    'direction': direction,
                    'angle': stored_direction['angle'],
                    'distance': stored_direction['distance'],
                    'target_frame': next_kf,
                    'status': 'at_keyframe',
                    'command': command
                }
            
            # Fallback: calculate dynamically towards destination
            all_kf_ids = sorted(self.all_keyframes.keys())
            try:
                current_idx = all_kf_ids.index(current_kf_id)
                
                # Check if reached or passed destination
                if self.destination_kf_id is not None:
                    if current_kf_id >= self.destination_kf_id:
                        return {'direction': 'stop', 'angle': 0, 'distance': 0,
                                'target_frame': self.destination_kf_id,
                                'status': 'destination_reached', 'command': 'stop'}
                
                # Move to next keyframe towards destination
                if current_idx + 1 < len(all_kf_ids):
                    next_kf_id = all_kf_ids[current_idx + 1]
                    
                    # Don't go past destination
                    if self.destination_kf_id is not None and next_kf_id > self.destination_kf_id:
                        return {'direction': 'stop', 'angle': 0, 'distance': 0,
                                'target_frame': self.destination_kf_id,
                                'status': 'destination_reached', 'command': 'stop'}
                    
                    next_kf = self.all_keyframes[next_kf_id]
                    next_x, next_y, _ = next_kf.pose
                    target_position = np.array([next_x, next_y])
                    target_distance = np.linalg.norm(target_position - current_position)
                    
                    # Calculate direction to next keyframe
                    rel_pos = target_position - current_position
                    required_yaw = np.arctan2(rel_pos[1], rel_pos[0])
                    angle_diff = np.degrees(required_yaw - current_yaw)
                    angle_diff = (angle_diff + 180) % 360 - 180
                    
                    # Determine action
                    if abs(angle_diff) > 15:  # Need to rotate first
                        if angle_diff > 0:
                            direction = 'rotate_left'
                            command = 'left'
                        else:
                            direction = 'rotate_right'
                            command = 'right'
                        return {
                            'direction': direction,
                            'angle': self._snap_angle(abs(angle_diff)),
                            'distance': target_distance,
                            'target_frame': next_kf_id,
                            'status': 'at_keyframe',
                            'command': command
                        }
                    else:  # Aligned, move forward
                        return {
                            'direction': 'forward',
                            'angle': 0,
                            'distance': target_distance,
                            'target_frame': next_kf_id,
                            'status': 'at_keyframe',
                            'command': 'forward'
                        }
                else:
                    # Last keyframe reached
                    return {
                        'direction': 'stop',
                        'angle': 0,
                        'distance': 0,
                        'target_frame': current_kf_id,
                        'status': 'destination_reached',
                        'command': 'stop'
                    }
            except ValueError:
                pass
        
        # Navigate to nearest keyframe
        nearest_kf = self.all_keyframes[current_kf_id]
        nearest_x, nearest_y, _ = nearest_kf.pose
        target_position = np.array([nearest_x, nearest_y])
        
        # Calculate direction
        rel_pos = target_position - current_position
        required_yaw = np.arctan2(rel_pos[1], rel_pos[0])
        angle_diff = np.degrees(required_yaw - current_yaw)
        angle_diff = (angle_diff + 180) % 360 - 180
        
        # Determine action
        if abs(angle_diff) > 15:  # Need to rotate first
            if angle_diff > 0:
                direction = 'rotate_left'
                command = 'left'
            else:
                direction = 'rotate_right'
                command = 'right'
            return {
                'direction': direction,
                'angle': self._snap_angle(abs(angle_diff)),
                'distance': min_dist,
                'target_frame': current_kf_id,
                'status': 'navigating',
                'command': command
            }
        else:  # Aligned, move forward
            return {
                'direction': 'forward',
                'angle': 0,
                'distance': min_dist,
                'target_frame': current_kf_id,
                'status': 'navigating',
                'command': 'forward'
            }

    def draw_navigation_overlay(self, frame, nav_direction):
        """Draw navigation direction overlay on frame"""
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        overlay = display_frame.copy()
        
        # Draw direction info box
        box_height = 120
        cv2.rectangle(overlay, (10, 10), (400, box_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
        
        # Direction text
        direction = nav_direction['direction']
        angle = nav_direction['angle']
        distance = nav_direction['distance']
        status = nav_direction.get('status', 'unknown')
        
        # Color coding
        if status == 'searching_forward':
            color = (0, 165, 255)  # Orange - searching by moving forward
            main_text = "SEARCHING - Moving Forward"
        elif status == 'searching_rotate':
            color = (0, 165, 255)  # Orange - searching by rotating
            main_text = f"SEARCHING - Rotating Right {angle}°"
        elif status == 'destination_reached':
            color = (0, 255, 0)  # Green
            main_text = "DESTINATION REACHED - STOP"
        elif status == 'at_keyframe':
            color = (0, 255, 255)  # Yellow
            if direction in ['rotate_left', 'forward_left']:
                main_text = f"At Keyframe - ROTATE LEFT {angle}°"
            elif direction in ['rotate_right', 'forward_right']:
                main_text = f"At Keyframe - ROTATE RIGHT {angle}°"
            else:
                main_text = f"At Keyframe - Next: {direction.upper()}"
        else:
            color = (255, 255, 255)  # White
            if direction == 'forward':
                main_text = "MOVE FORWARD"
            elif direction == 'rotate_left' or direction == 'forward_left':
                main_text = f"ROTATE LEFT {angle}°"
            elif direction == 'rotate_right' or direction == 'forward_right':
                main_text = f"ROTATE RIGHT {angle}°"
            elif direction == 'stop':
                main_text = "STOP"
                color = (0, 0, 255)  # Red
            else:
                main_text = "CALCULATING..."
        
        # Draw text
        cv2.putText(display_frame, main_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)
        cv2.putText(display_frame, f"Distance: {distance:.2f}m", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        if nav_direction['target_frame'] is not None:
            cv2.putText(display_frame, f"Target: Frame {nav_direction['target_frame']}", (20, 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw direction arrow
        arrow_center_x = w - 100
        arrow_center_y = 80
        arrow_length = 50
        
        if direction == 'forward' or direction.startswith('forward'):
            # Up arrow
            cv2.arrowedLine(display_frame, 
                          (arrow_center_x, arrow_center_y + arrow_length//2),
                          (arrow_center_x, arrow_center_y - arrow_length//2),
                          color, 3, tipLength=0.3)
            if 'left' in direction:
                cv2.arrowedLine(display_frame,
                              (arrow_center_x - 20, arrow_center_y),
                              (arrow_center_x - 40, arrow_center_y - 20),
                              color, 2, tipLength=0.3)
            elif 'right' in direction:
                cv2.arrowedLine(display_frame,
                              (arrow_center_x + 20, arrow_center_y),
                              (arrow_center_x + 40, arrow_center_y - 20),
                              color, 2, tipLength=0.3)
        elif direction == 'rotate_left':
            # Left curved arrow
            cv2.ellipse(display_frame, (arrow_center_x, arrow_center_y),
                       (30, 30), 0, 180, 90, color, 3)
            cv2.arrowedLine(display_frame,
                          (arrow_center_x - 25, arrow_center_y - 5),
                          (arrow_center_x - 35, arrow_center_y + 5),
                          color, 3, tipLength=0.5)
        elif direction == 'rotate_right':
            # Right curved arrow
            cv2.ellipse(display_frame, (arrow_center_x, arrow_center_y),
                       (30, 30), 0, 0, -90, color, 3)
            cv2.arrowedLine(display_frame,
                          (arrow_center_x + 25, arrow_center_y - 5),
                          (arrow_center_x + 35, arrow_center_y + 5),
                          color, 3, tipLength=0.5)
        elif direction == 'stop':
            # Stop sign
            cv2.circle(display_frame, (arrow_center_x, arrow_center_y), 35, color, 3)
            cv2.putText(display_frame, "STOP", (arrow_center_x - 25, arrow_center_y + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 3)
        
        return display_frame
    
    def print_navigation_command(self, nav_direction):
        """Print navigation command to terminal"""
        direction = nav_direction['direction']
        angle = nav_direction['angle']
        distance = nav_direction['distance']
        status = nav_direction.get('status', 'unknown')
        target_frame = nav_direction.get('target_frame', 'N/A')
        
        # Build command text based on status and direction
        if status == 'searching_forward':
            command = "🔍 SEARCHING - Moving forward to find features"
            symbol = "🔎"
        elif status == 'searching_rotate':
            command = f"🔍 SEARCHING - Rotating right {angle}° to scan area"
            symbol = "🔄"
        elif status == 'stopping_for_rotation':
            command = "🛑 STOP - Preparing for rotation"
            symbol = "⏸️"
        elif status == 'destination_reached':
            command = "🟢 DESTINATION REACHED - STOP"
            symbol = "⏹️"
        elif status == 'at_keyframe':
            symbol = "📍"
            if direction == 'forward':
                command = "⬆️  MOVE FORWARD"
            elif direction == 'rotate_left' or direction == 'forward_left':
                command = f"↶  ROTATE LEFT {angle}° (STOP → ROTATE → 2sec DELAY)"
            elif direction == 'rotate_right' or direction == 'forward_right':
                command = f"↷  ROTATE RIGHT {angle}° (STOP → ROTATE → 2sec DELAY)"
            elif direction == 'stop':
                command = "⏹️  STOP"
            else:
                command = "⏳ CALCULATING..."
        else:  # navigating
            symbol = "🧭"
            if direction == 'forward':
                command = "⬆️  MOVE FORWARD"
            elif direction == 'rotate_left' or direction == 'forward_left':
                command = f"↶  ROTATE LEFT {angle}° (STOP → ROTATE → 2sec DELAY)"
            elif direction == 'rotate_right' or direction == 'forward_right':
                command = f"↷  ROTATE RIGHT {angle}° (STOP → ROTATE → 2sec DELAY)"
            elif direction == 'stop':
                command = "⏹️  STOP"
            else:
                command = "⏳ CALCULATING..."
        
        # Print formatted command
        print(f"   {symbol} {command} | Target: KF#{target_frame} | Distance: {distance:.2f}m")

    def draw_map_with_live_position(self, current_frame=None):
        """Draw top-down map with live rover position"""
        canvas = np.zeros((self.map_canvas_size, self.map_canvas_size, 3), dtype=np.uint8)
        center_x, center_y = self.map_canvas_size // 2, self.map_canvas_size // 2
        
        # Calculate map bounds
        if self.trajectory:
            all_x = [p[0] for p in self.trajectory]
            all_y = [p[1] for p in self.trajectory]
            map_center_x = (max(all_x) + min(all_x)) / 2
            map_center_y = (max(all_y) + min(all_y)) / 2
        else:
            map_center_x, map_center_y = 0, 0

        def project(x, y):
            sx = int(center_x + (x - map_center_x) * self.map_scale)
            sy = int(center_y - (y - map_center_y) * self.map_scale)
            return sx, sy

        # Draw map features (spatial grid)
        for feat in self.spatial_grid.values():
            sx, sy = project(feat['x'], feat['y'])
            if 0 <= sx < self.map_canvas_size and 0 <= sy < self.map_canvas_size:
                brightness = int(np.clip(200.0 / (feat['z'] + 0.5), 40, 255))
                cv2.circle(canvas, (sx, sy), 2, (brightness, brightness, brightness), -1)

        # Draw keyframes
        for kf_id, kf in self.all_keyframes.items():
            kf_x, kf_y, kf_yaw = kf.pose
            kf_sx, kf_sy = project(kf_x, kf_y)
            if 0 <= kf_sx < self.map_canvas_size and 0 <= kf_sy < self.map_canvas_size:
                # Draw keyframe as line
                perp_angle = kf_yaw + math.pi / 2
                line_len = 30
                line_ex = int(kf_sx + line_len * math.cos(perp_angle))
                line_ey = int(kf_sy - line_len * math.sin(perp_angle))
                cv2.line(canvas, (kf_sx, kf_sy), (line_ex, line_ey), (255, 0, 0), 3, cv2.LINE_AA)
                
                # Highlight if current match
                if self.current_kf_id == kf_id:
                    cv2.circle(canvas, (kf_sx, kf_sy), 12, (0, 255, 255), 3)
                    cv2.putText(canvas, f"KF{kf_id}*", (kf_sx - 60, kf_sy - 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    cv2.putText(canvas, f"KF{kf_id}", (kf_sx + 8, kf_sy - 8),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Draw trajectory
        if len(self.trajectory) > 1:
            for i in range(len(self.trajectory) - 1):
                p1 = project(self.trajectory[i][0], self.trajectory[i][1])
                p2 = project(self.trajectory[i+1][0], self.trajectory[i+1][1])
                if (0 <= p1[0] < self.map_canvas_size and 0 <= p1[1] < self.map_canvas_size and
                    0 <= p2[0] < self.map_canvas_size and 0 <= p2[1] < self.map_canvas_size):
                    cv2.line(canvas, p1, p2, (0, 255, 0), 2)

        # Draw current rover position
        if self.current_pose is not None:
            rover_x, rover_y, rover_yaw = self.current_pose
            rover_sx, rover_sy = project(rover_x, rover_y)
            
            if 0 <= rover_sx < self.map_canvas_size and 0 <= rover_sy < self.map_canvas_size:
                # Draw rover circle
                cv2.circle(canvas, (rover_sx, rover_sy), 15, (0, 0, 255), -1)
                
                # Draw heading arrow
                arrow_len = 50
                arrow_ex = int(rover_sx + arrow_len * math.cos(rover_yaw))
                arrow_ey = int(rover_sy - arrow_len * math.sin(rover_yaw))
                cv2.arrowedLine(canvas, (rover_sx, rover_sy), (arrow_ex, arrow_ey), 
                               (0, 0, 255), 3, tipLength=0.3)
                
                # Label
                cv2.putText(canvas, "ROVER", (rover_sx - 50, rover_sy - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Info overlay
        cv2.putText(canvas, "LIVE ROVER NAVIGATION", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, f"Distance: {self.total_distance:.3f}m", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(canvas, f"Current KF: {self.current_kf_id if self.current_kf_id else 'N/A'}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return canvas

    def run_live_navigation(self, video_path_or_source, process_every_n=5):
        """
        Run live navigation with real-time localization and visualization
        
        Args:
            video_path_or_source: Video file path, camera index (0, 1), or 'rover' for rover stream
            process_every_n: Process every N frames for efficiency
        """
        print(f"\n{'='*80}")
        print("🚀 STARTING LIVE NAVIGATION")
        print(f"{'='*80}\n")
        
        # Handle different video sources
        if video_path_or_source == 'rover' or self.use_rover_stream:
            if self.rover_stream_url:
                print(f"📹 Using rover stream: {self.rover_stream_url}")
                cap = cv2.VideoCapture(self.rover_stream_url)
            else:
                print("❌ Rover stream not configured")
                return
        elif isinstance(video_path_or_source, int):
            print(f"📹 Using camera index: {video_path_or_source}")
            cap = cv2.VideoCapture(video_path_or_source)
        else:
            print(f"📹 Using video file: {video_path_or_source}")
            cap = cv2.VideoCapture(video_path_or_source)
        
        if not cap.isOpened():
            print(f"❌ Cannot open video source: {video_path_or_source}")
            return
        
        frame_count = 0
        emergency_stop_triggered = False
        
        print("\n⚠️  CONTROLS:")
        print("   Press 'Q' to EMERGENCY STOP and exit")
        print("   Press 'R' to resume after emergency stop")
        print(f"{'='*80}\n")
        
        while True:
            # REAL-TIME FRAME CAPTURE: Clear buffer and get latest frame
            # This prevents frame stacking/queuing for true real-time navigation
            for _ in range(2):  # Grab and discard buffered frames
                cap.grab()
            
            ret, frame = cap.retrieve()
            if not ret:
                # Try regular read if retrieve fails
                ret, frame = cap.read()
                if not ret:
                    print("\n⚠️ End of video stream or read error")
                    break
            
            frame = cv2.resize(frame, (640, 480))
            frame_count += 1
            
            # Check for emergency stop key FIRST - highest priority
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                if not emergency_stop_triggered:
                    emergency_stop_triggered = True
                    self.rover_controller.emergency_stop_now()
                    print("\n🚨 EMERGENCY STOP - Press 'Q' again to exit or 'R' to resume")
                else:
                    print("\n⏹️ Exiting navigation...")
                    break
            elif key == ord('r') or key == ord('R'):
                if emergency_stop_triggered:
                    emergency_stop_triggered = False
                    self.rover_controller.reset_emergency_stop()
                    print("\n✅ Resuming navigation...")
            elif key == 27:  # ESC key
                print("\n⏹️ Navigation stopped by user (ESC)")
                self.rover_controller.emergency_stop_now()
                break
            
            # Skip processing if emergency stop is active
            if emergency_stop_triggered:
                # Show emergency stop overlay
                emergency_frame = frame.copy()
                cv2.rectangle(emergency_frame, (0, 0), (640, 100), (0, 0, 255), -1)
                cv2.putText(emergency_frame, "EMERGENCY STOP ACTIVE", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                cv2.putText(emergency_frame, "Press 'R' to resume | 'Q' to exit", (80, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                map_canvas = self.draw_map_with_live_position(frame)
                cv2.imshow("Navigation Map", map_canvas)
                cv2.imshow("Current Frame", emergency_frame)
                continue
            
            # Skip processing if rotation is in progress (navigation paused during rotation)
            if self.rover_controller.is_rotating:
                # Show rotation overlay
                rotation_frame = frame.copy()
                cv2.rectangle(rotation_frame, (0, 0), (640, 100), (0, 165, 255), -1)
                cv2.putText(rotation_frame, "ROTATION IN PROGRESS", (120, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                cv2.putText(rotation_frame, "Navigation Paused...", (180, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                map_canvas = self.draw_map_with_live_position(frame)
                cv2.imshow("Navigation Map", map_canvas)
                cv2.imshow("Current Frame", rotation_frame)
                continue
            
            # Process EVERY frame in real-time (no skipping for low latency)
            # Localize current frame
            start_time = time.time()
            matched_kf_id, confidence, inliers, estimated_pose = self.localize_frame(frame)
            elapsed = time.time() - start_time
            
            # Update pose and trajectory if localized
            if matched_kf_id is not None and estimated_pose is not None:
                # Reset lost frame counter when successfully localized
                self.lost_frame_count = 0
                
                # Update current pose
                old_pose = self.current_pose
                self.current_pose = estimated_pose
                self.current_kf_id = matched_kf_id
                
                # Update trajectory and distance
                if old_pose is not None:
                    dx = self.current_pose[0] - old_pose[0]
                    dy = self.current_pose[1] - old_pose[1]
                    distance = np.sqrt(dx**2 + dy**2)
                    self.total_distance += distance
                
                self.trajectory.append((self.current_pose[0], self.current_pose[1]))
                
                print(f"[Frame {frame_count:5d}] ✅ KF#{matched_kf_id} | "
                      f"Conf: {confidence*100:.1f}% | Inliers: {inliers} | "
                      f"Time: {elapsed*1000:.1f}ms | Dist: {self.total_distance:.3f}m")
            else:
                # No match found - increment lost counter
                self.lost_frame_count += 1
                print(f"[Frame {frame_count:5d}] ❌ No match found (Lost frames: {self.lost_frame_count})")
            
            # Get navigation direction
            nav_direction = self.get_navigation_direction()
            
            # If no localization, just keep going forward (unless navigation says turn right)
            if matched_kf_id is None:
                # Check if the calculated navigation direction is to turn right
                if nav_direction.get('command') == 'right':
                    # Navigation wants right turn, so execute it
                    nav_direction = {
                        'direction': 'rotate_right', 
                        'angle': 90,
                        'distance': 0,
                        'target_frame': None,
                        'status': 'searching_rotate',
                        'command': 'right'
                    }
                else:
                    # Default: keep moving forward to search for features
                    nav_direction = {
                        'direction': 'forward',
                        'angle': 0,
                        'distance': 0.3,
                        'target_frame': None,
                        'status': 'searching_forward',
                        'command': 'forward'
                    }
            
            # Print navigation direction to terminal
            self.print_navigation_command(nav_direction)
            
            # Execute rover command if auto-control is enabled (timing handled in RoverController)
            if self.auto_control:
                self.rover_controller.execute_navigation_command(nav_direction)
            
            # Check if destination reached - send stop and exit loop
            if nav_direction.get('status') == 'destination_reached':
                if self.auto_control:
                    print("\n🛑 Destination reached - Sending STOP command to rover...")
                    self.rover_controller.send_command('stop')
                # Show final frame then exit
                frame_with_nav = self.draw_navigation_overlay(frame, nav_direction)
                map_canvas = self.draw_map_with_live_position(frame)
                cv2.imshow("Navigation Map", map_canvas)
                cv2.imshow("Current Frame", frame_with_nav)
                cv2.waitKey(2000)  # Show final state for 2 seconds
                break  # Exit navigation loop
            
            # Visualization with navigation overlay
            frame_with_nav = self.draw_navigation_overlay(frame, nav_direction)
            map_canvas = self.draw_map_with_live_position(frame)
            cv2.imshow("Navigation Map", map_canvas)
            cv2.imshow("Current Frame", frame_with_nav)
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Send final stop command
        if self.auto_control:
            print("\n🛑 Sending final stop command to rover...")
            self.rover_controller.send_command('stop')
        
        print(f"\n{'='*80}")
        print("🎯 NAVIGATION COMPLETE")
        print(f"{'='*80}")
        print(f"Total Distance: {self.total_distance:.3f}m")
        if self.current_pose is not None:
            print(f"Final Position: ({self.current_pose[0]:.3f}, {self.current_pose[1]:.3f})")
        print(f"Frames Processed: {frame_count}")
        print(f"{'='*80}\n")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*80)
    print("🚀 LIVE ROVER NAVIGATION WITH REAL-TIME LOCALIZATION")
    print("="*80 + "\n")
    
    # Ask for control mode
    print("🎮 Control Mode:")
    print("  1. Simulation (display commands only)")
    print("  2. Automatic Rover Control (send commands to rover)")
    mode = input("Select mode (1/2, default 1): ").strip()
    
    enable_rover = (mode == '2')
    
    if enable_rover:
        print("\n🤖 AUTOMATIC ROVER CONTROL ENABLED")
        rover_ip = input("🌐 Rover IP address (default 10.226.57.127): ").strip()
        rover_ip = rover_ip if rover_ip else "10.226.57.127"
        rover_port = input("🔌 Rover port (default 8080): ").strip()
        rover_port = int(rover_port) if rover_port else 8080
    else:
        print("\n🔒 SIMULATION MODE - Commands will be displayed only")
        rover_ip = "10.226.57.127"
        rover_port = 8080
    
    # Ask for video source
    print("\n📹 Video Source:")
    print("  1. Video file")
    print("  2. Webcam/USB camera")
    print("  3. Live rover stream")
    source_type = input("Select source (1/2/3, default 1): ").strip()
    
    use_rover_stream = False
    video_source = None
    
    if source_type == '2':
        camera_idx = input("📷 Camera index (default 0): ").strip()
        video_source = int(camera_idx) if camera_idx else 0
        print(f"✅ Using camera index: {video_source}")
    elif source_type == '3':
        use_rover_stream = True
        video_source = 'rover'
        print(f"✅ Using live rover stream from {rover_ip}:{rover_port}")
    else:
        video_source = input("\n📹 Enter video file path: ").strip()
        if not os.path.exists(video_source):
            print(f"❌ Video file not found: {video_source}")
            return
        print(f"✅ Using video file: {video_source}")
    
    # Initialize system with rover control
    nav = LiveNavigationSystem(
        fx=600.0, fy=600.0, cx=320.0, cy=240.0,
        enable_rover_control=enable_rover,
        rover_ip=rover_ip,
        rover_port=rover_port,
        use_rover_stream=use_rover_stream
    )
    
    # Load map
    storage_dir = "keyframes_storage"
    if not nav.load_map(storage_dir):
        print("❌ Failed to load map")
        return
    
    # Show available keyframes and ask for destination
    all_kf_ids = sorted(nav.all_keyframes.keys())
    print(f"\n📍 Available Keyframes: {all_kf_ids}")
    print(f"   Total: {len(all_kf_ids)} keyframes (from KF#{all_kf_ids[0]} to KF#{all_kf_ids[-1]})")
    
    # Ask for destination
    print(f"\n🎯 Select Destination Keyframe:")
    dest_input = input(f"   Enter keyframe number (e.g., {all_kf_ids[-1]} for last): ").strip()
    
    if dest_input:
        try:
            destination_kf = int(dest_input)
            if destination_kf in all_kf_ids:
                nav.destination_kf_id = destination_kf
                print(f"   ✅ Destination set to Keyframe #{destination_kf}")
            else:
                print(f"   ⚠️  Invalid keyframe. Using last keyframe: #{all_kf_ids[-1]}")
                nav.destination_kf_id = all_kf_ids[-1]
        except ValueError:
            print(f"   ⚠️  Invalid input. Using last keyframe: #{all_kf_ids[-1]}")
            nav.destination_kf_id = all_kf_ids[-1]
    else:
        nav.destination_kf_id = all_kf_ids[-1]
        print(f"   ℹ️  No input. Using last keyframe: #{all_kf_ids[-1]}")
    
    # Run navigation
    process_every = input("\n⚙️ Process every N frames (default 5): ").strip()
    process_every = int(process_every) if process_every else 5
    
    print("\n" + "="*80)
    print("⚠️  NAVIGATION CONTROLS:")
    print("   Press 'Q' to EMERGENCY STOP (press twice to exit)")
    print("   Press 'R' to RESUME after emergency stop")
    print("   Press 'ESC' to stop and exit immediately")
    if enable_rover:
        print("\n⚠️  ROVER WILL MOVE AUTOMATICALLY")
    else:
        print("\nℹ️  Simulation mode - Commands displayed only")
    print("="*80 + "\n")
    
    nav.run_live_navigation(video_source, process_every_n=process_every)


if __name__ == "__main__":
    main()