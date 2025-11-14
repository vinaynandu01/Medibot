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
# ROVER CONTROL
# ═══════════════════════════════════════════════════════════════════════════

class RoverController:
    """Controls the actual rover hardware via HTTP commands"""
    
    def __init__(self, rover_ip="10.226.57.127", rover_port=8080, enable_control=True):
        self.rover_ip = rover_ip
        self.rover_port = rover_port
        self.enable_control = enable_control
        self.last_command_time = 0
        self.command_cooldown = 0.3
        self.emergency_stop = False
        self.last_command_sent = None
        self.is_moving_forward = False
        
        # SIMPLIFIED ROTATION STATE MACHINE
        self.rotation_state = None  # None, 'pre_stop', 'rotating', 'post_stop', 'cooldown'
        self.rotation_state_start = 0
        self.rotation_direction = None  # 'left' or 'right'
        
    def send_command(self, command):
        """Send command to rover"""
        if not self.enable_control:
            print(f"🔒 [SIMULATION] Command: {command}")
            return True
        
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
    
    def update_rotation_state_machine(self):
        """
        Handle rotation sequence state machine.
        Returns: True if rotation is in progress (block navigation), False otherwise
        """
        if self.rotation_state is None:
            return False  # Not rotating
        
        current_time = time.time()
        elapsed = current_time - self.rotation_state_start
        
        if self.rotation_state == 'pre_stop':
            # State 1: Stop before rotation (1 second)
            if elapsed >= 1.0:
                print(f"🔄 Sending {self.rotation_direction.upper()} rotation command (ONCE)...")
                self.send_command(self.rotation_direction)
                self.rotation_state = 'rotating'
                self.rotation_state_start = current_time
                return True
            else:
                remaining = 1.0 - elapsed
                if remaining > 0.8:  # Only print at start
                    print(f"🛑 Pre-rotation stop: {remaining:.1f}s remaining...")
                return True
        
        elif self.rotation_state == 'rotating':
            # State 2: Rotating (5 seconds)
            if elapsed >= 5.0:
                print(f"🛑 Stopping rotation...")
                self.send_command('stop')
                self.rotation_state = 'post_stop'
                self.rotation_state_start = current_time
                return True
            else:
                return True  # Still rotating
        
        elif self.rotation_state == 'post_stop':
            # State 3: Post-rotation stabilization (1 second)
            if elapsed >= 1.0:
                print(f"✅ Rotation complete - preparing fresh navigation...")
                self.rotation_state = 'cooldown'
                self.rotation_state_start = current_time
                return True
            else:
                return True  # Still stabilizing
        
        elif self.rotation_state == 'cooldown':
            # State 4: Frame clearing cooldown (0.5 seconds)
            if elapsed >= 0.5:
                print(f"🆕 Fresh navigation ready - resuming...")
                self.rotation_state = None
                self.rotation_direction = None
                return False  # Resume navigation
            else:
                return True  # Still in cooldown
        
        return False
    
    def start_rotation(self, direction):
        """Initiate rotation sequence (call ONCE per rotation)"""
        if self.rotation_state is not None:
            return False  # Already rotating, ignore
        
        print(f"\n{'='*60}")
        print(f"🔄 ROTATION SEQUENCE: {direction.upper()}")
        print(f"{'='*60}")
        
        self.rotation_state = 'pre_stop'
        self.rotation_state_start = time.time()
        self.rotation_direction = direction
        self.is_moving_forward = False
        
        # Send initial stop
        print(f"🛑 Step 1: Stopping rover...")
        self.send_command('stop')
        return True
    
    def execute_navigation_command(self, nav_direction, current_kf_id=None):
        """Execute navigation command with proper rotation handling"""
        # Check if rotation is in progress
        if self.update_rotation_state_machine():
            return False  # Block navigation during rotation
        
        if self.emergency_stop:
            return False
        
        current_time = time.time()
        command = nav_direction.get('command', 'unknown')
        
        if command == 'forward':
            if not self.is_moving_forward:
                success = self.send_command('forward')
                if success:
                    self.last_command_sent = 'forward'
                    self.last_command_time = current_time
                    self.is_moving_forward = True
                return success
            else:
                return True  # Already moving forward
            
        elif command in ['left', 'right']:
            # Start rotation sequence (only if not already rotating)
            if self.rotation_state is None:
                return self.start_rotation(command)
            else:
                return False  # Already rotating
            
        elif command == 'stop':
            success = self.send_command('stop')
            if success:
                self.last_command_sent = 'stop'
                self.last_command_time = current_time
                self.is_moving_forward = False
            return success
        
        return False
    
    def emergency_stop_now(self):
        """Activate emergency stop immediately"""
        self.emergency_stop = True
        self.is_moving_forward = False
        self.rotation_state = None  # Cancel any rotation
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
# LIVE NAVIGATION SYSTEM
# ═══════════════════════════════════════════════════════════════════════════

class LiveNavigationSystem:
    def __init__(self, fx=600.0, fy=600.0, cx=320.0, cy=240.0, 
                 enable_rover_control=False, rover_ip="10.226.57.127", rover_port=8080,
                 use_rover_stream=False, destination_kf_id=None):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
        self.storage_manager = KeyframeStorageManager()
        
        # ORB for feature extraction
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.orb = cv2.ORB_create(nfeatures=500)
        
        # Map data
        self.all_keyframes = {}
        self.spatial_grid = {}
        self.keyframe_directions = {}
        
        # Current state
        self.current_pose = None
        self.current_kf_id = None
        self.trajectory = []
        self.total_distance = 0.0
        
        # Navigation target
        self.destination_kf_id = destination_kf_id
        self.start_kf_id = None
        self.destination_reached = False
        
        # Rover control integration
        self.rover_controller = RoverController(
            rover_ip=rover_ip, 
            rover_port=rover_port, 
            enable_control=enable_rover_control
        )
        self.auto_control = enable_rover_control
        
        # Rover stream settings
        self.use_rover_stream = use_rover_stream
        self.rover_stream_url = f"http://{rover_ip}:{rover_port}/video_feed" if use_rover_stream else None
        
        # Visualization
        self.map_canvas_size = 900
        self.map_scale = 100

    def feature_extraction(self, gray):
        """Extract ORB features from grayscale image"""
        enhanced = self.clahe.apply(gray)
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=5)
        kp, desc = self.orb.detectAndCompute(enhanced, None)
        if kp is None or desc is None or len(kp) == 0:
            return [], None
        
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
                
                for x_world, y_world, z_depth in kf.features_3d:
                    key = self._spatial_grid_key(x_world, y_world)
                    if key not in self.spatial_grid:
                        self.spatial_grid[key] = {
                            'x': x_world, 'y': y_world, 'z': z_depth
                        }
        
        print(f"✅ Loaded {len(self.all_keyframes)} keyframes")
        self._load_direction_map(storage_dir)
        return True
    
    def _load_direction_map(self, storage_dir):
        """Load navigation direction map from JSON file"""
        try:
            direction_map_file = os.path.join(storage_dir, "direction_map.json")
            if os.path.exists(direction_map_file):
                with open(direction_map_file, 'r') as f:
                    data = json.load(f)
                    self.keyframe_directions = {int(k): v for k, v in data.get('keyframe_directions', {}).items()}
                print(f"✅ Loaded direction map with {len(self.keyframe_directions)} entries")
            else:
                print("⚠️ No direction map found - will calculate on-the-fly")
        except Exception as e:
            print(f"⚠️ Could not load direction map: {e}")

    def _spatial_grid_key(self, x, y):
        return (int(x / 0.10), int(y / 0.10))

    def localize_frame(self, frame):
        """Localize current frame against all keyframes"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp_test, desc_test = self.feature_extraction(gray)
        
        if desc_test is None or len(kp_test) == 0:
            return None, 0.0, 0, None

        best_match_kf_id = None
        best_confidence = 0.0
        best_num_matches = 0
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

            good_matches = []
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            if len(good_matches) < 10:
                continue

            match_confidence = len(good_matches) / (len(kf.descriptors) + 1e-6)

            if match_confidence < 0.10:
                continue
            
            if (match_confidence > best_confidence) or (
                abs(match_confidence - best_confidence) < 1e-4 and len(good_matches) > best_num_matches
            ):
                best_confidence = match_confidence
                best_num_matches = len(good_matches)
                best_match_kf_id = kf_id
                best_pose = kf.pose.copy()

        return best_match_kf_id, best_confidence, best_num_matches, best_pose

    def _snap_angle(self, angle):
        """Snap angle to nearest standard value"""
        if angle < 22.5:
            return 0
        elif angle < 67.5:
            return 45
        elif angle < 135:
            return 90
        else:
            return 180

    def get_navigation_direction(self):
        """Get direction to move from current position towards destination"""
        if self.current_pose is None or len(self.all_keyframes) == 0:
            return {'direction': 'stop', 'angle': 0, 'distance': 0, 'target_frame': None, 
                    'status': 'no_pose', 'command': 'stop'}
        
        if self.destination_kf_id is not None and self.current_kf_id == self.destination_kf_id:
            if not self.destination_reached:
                self.destination_reached = True
                print(f"\n{'='*80}")
                print(f"🎯 DESTINATION REACHED! Keyframe #{self.destination_kf_id}")
                print(f"{'='*80}\n")
            return {'direction': 'stop', 'angle': 0, 'distance': 0, 
                    'target_frame': self.destination_kf_id, 
                    'status': 'destination_reached', 'command': 'stop'}
        
        current_x, current_y, current_yaw = self.current_pose
        current_position = np.array([current_x, current_y])
        
        # Find nearest keyframe
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
        
        if self.start_kf_id is None:
            self.start_kf_id = current_kf_id
            print(f"\n📍 Start: KF#{self.start_kf_id} → Destination: KF#{self.destination_kf_id}\n")
        
        # Use pre-calculated direction if close to keyframe
        if min_dist < 0.5:
            if current_kf_id in self.keyframe_directions:
                stored_direction = self.keyframe_directions[current_kf_id]
                next_kf = stored_direction.get('target_frame', current_kf_id + 1)
                
                if self.destination_kf_id is not None:
                    if current_kf_id >= self.destination_kf_id:
                        return {'direction': 'stop', 'angle': 0, 'distance': 0, 
                                'target_frame': self.destination_kf_id,
                                'status': 'destination_reached', 'command': 'stop'}
                    
                    if next_kf > self.destination_kf_id:
                        return {'direction': 'stop', 'angle': 0, 'distance': 0, 
                                'target_frame': self.destination_kf_id,
                                'status': 'destination_reached', 'command': 'stop'}
                
                direction = stored_direction['direction']
                
                if direction in ['rotate_left', 'forward_left', 'rotate_right', 'forward_right']:
                    command = 'left' if direction in ['rotate_left', 'forward_left'] else 'right'
                    return {
                        'direction': direction,
                        'angle': stored_direction['angle'],
                        'distance': min_dist,
                        'target_frame': next_kf,
                        'status': 'at_keyframe',
                        'command': command,
                        'at_rotation_keyframe': min_dist < 0.25
                    }
                else:
                    command = direction
                
                return {
                    'direction': direction,
                    'angle': stored_direction['angle'],
                    'distance': stored_direction['distance'],
                    'target_frame': next_kf,
                    'status': 'at_keyframe',
                    'command': command
                }
        
        # Navigate to nearest keyframe
        nearest_kf = self.all_keyframes[current_kf_id]
        nearest_x, nearest_y, _ = nearest_kf.pose
        target_position = np.array([nearest_x, nearest_y])
        
        rel_pos = target_position - current_position
        required_yaw = np.arctan2(rel_pos[1], rel_pos[0])
        angle_diff = np.degrees(required_yaw - current_yaw)
        angle_diff = (angle_diff + 180) % 360 - 180
        
        if abs(angle_diff) > 15:
            direction = 'rotate_left' if angle_diff > 0 else 'rotate_right'
            command = 'left' if angle_diff > 0 else 'right'
            return {
                'direction': direction,
                'angle': self._snap_angle(abs(angle_diff)),
                'distance': min_dist,
                'target_frame': current_kf_id,
                'status': 'navigating',
                'command': command
            }
        else:
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
        
        box_height = 120
        cv2.rectangle(overlay, (10, 10), (400, box_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
        
        direction = nav_direction['direction']
        angle = nav_direction['angle']
        distance = nav_direction['distance']
        status = nav_direction.get('status', 'unknown')
        
        if status == 'destination_reached':
            color = (0, 255, 0)
            main_text = "DESTINATION REACHED"
        elif status == 'at_keyframe':
            color = (0, 255, 255)
            if direction in ['rotate_left', 'forward_left']:
                main_text = f"ROTATE LEFT {angle}°"
            elif direction in ['rotate_right', 'forward_right']:
                main_text = f"ROTATE RIGHT {angle}°"
            else:
                main_text = f"Next: {direction.upper()}"
        else:
            color = (255, 255, 255)
            if direction == 'forward':
                main_text = "MOVE FORWARD"
            elif direction in ['rotate_left', 'forward_left']:
                main_text = f"ROTATE LEFT {angle}°"
            elif direction in ['rotate_right', 'forward_right']:
                main_text = f"ROTATE RIGHT {angle}°"
            elif direction == 'stop':
                main_text = "STOP"
                color = (0, 0, 255)
            else:
                main_text = "CALCULATING..."
        
        cv2.putText(display_frame, main_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)
        cv2.putText(display_frame, f"Distance: {distance:.2f}m", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        if nav_direction['target_frame'] is not None:
            cv2.putText(display_frame, f"Target: KF{nav_direction['target_frame']}", (20, 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Direction arrow
        arrow_center_x = w - 100
        arrow_center_y = 80
        arrow_length = 50
        
        if direction == 'forward' or direction.startswith('forward'):
            cv2.arrowedLine(display_frame, 
                          (arrow_center_x, arrow_center_y + arrow_length//2),
                          (arrow_center_x, arrow_center_y - arrow_length//2),
                          color, 3, tipLength=0.3)
        elif direction == 'rotate_left':
            cv2.ellipse(display_frame, (arrow_center_x, arrow_center_y),
                       (30, 30), 0, 180, 90, color, 3)
            cv2.arrowedLine(display_frame,
                          (arrow_center_x - 25, arrow_center_y - 5),
                          (arrow_center_x - 35, arrow_center_y + 5),
                          color, 3, tipLength=0.5)
        elif direction == 'rotate_right':
            cv2.ellipse(display_frame, (arrow_center_x, arrow_center_y),
                       (30, 30), 0, 0, -90, color, 3)
            cv2.arrowedLine(display_frame,
                          (arrow_center_x + 25, arrow_center_y - 5),
                          (arrow_center_x + 35, arrow_center_y + 5),
                          color, 3, tipLength=0.5)
        elif direction == 'stop':
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
        
        if status == 'destination_reached':
            command = "🟢 DESTINATION REACHED"
            symbol = "⏹️"
        elif status == 'at_keyframe':
            symbol = "📍"
            if direction == 'forward':
                command = "⬆️  FORWARD"
            elif direction in ['rotate_left', 'forward_left']:
                command = f"↶  LEFT {angle}°"
            elif direction in ['rotate_right', 'forward_right']:
                command = f"↷  RIGHT {angle}°"
            elif direction == 'stop':
                command = "⏹️  STOP"
            else:
                command = "⏳ CALCULATING"
        else:
            symbol = "🧭"
            if direction == 'forward':
                command = "⬆️  FORWARD"
            elif direction in ['rotate_left', 'forward_left']:
                command = f"↶  LEFT {angle}°"
            elif direction in ['rotate_right', 'forward_right']:
                command = f"↷  RIGHT {angle}°"
            elif direction == 'stop':
                command = "⏹️  STOP"
            else:
                command = "⏳ CALCULATING"
        
        print(f"   {symbol} {command} | Target: KF#{target_frame} | Dist: {distance:.2f}m")

    def draw_map_with_live_position(self, current_frame=None):
        """Draw top-down map with live rover position"""
        canvas = np.zeros((self.map_canvas_size, self.map_canvas_size, 3), dtype=np.uint8)
        center_x, center_y = self.map_canvas_size // 2, self.map_canvas_size // 2
        
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

        # Draw spatial grid
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
                perp_angle = kf_yaw + math.pi / 2
                line_len = 30
                line_ex = int(kf_sx + line_len * math.cos(perp_angle))
                line_ey = int(kf_sy - line_len * math.sin(perp_angle))
                cv2.line(canvas, (kf_sx, kf_sy), (line_ex, line_ey), (255, 0, 0), 3, cv2.LINE_AA)
                
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

        # Draw rover
        if self.current_pose is not None:
            rover_x, rover_y, rover_yaw = self.current_pose
            rover_sx, rover_sy = project(rover_x, rover_y)
            
            if 0 <= rover_sx < self.map_canvas_size and 0 <= rover_sy < self.map_canvas_size:
                cv2.circle(canvas, (rover_sx, rover_sy), 15, (0, 0, 255), -1)
                
                arrow_len = 50
                arrow_ex = int(rover_sx + arrow_len * math.cos(rover_yaw))
                arrow_ey = int(rover_sy - arrow_len * math.sin(rover_yaw))
                cv2.arrowedLine(canvas, (rover_sx, rover_sy), (arrow_ex, arrow_ey), 
                               (0, 0, 255), 3, tipLength=0.3)
                
                cv2.putText(canvas, "ROVER", (rover_sx - 50, rover_sy - 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Info overlay
        cv2.putText(canvas, "LIVE NAVIGATION", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(canvas, f"Distance: {self.total_distance:.3f}m", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(canvas, f"Current KF: {self.current_kf_id if self.current_kf_id else 'N/A'}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return canvas

    def run_live_navigation(self, video_path_or_source, process_every_n=5):
        """Run live navigation with real-time localization"""
        print(f"\n{'='*80}")
        print("🚀 STARTING LIVE NAVIGATION")
        print(f"{'='*80}\n")
        
        if video_path_or_source == 'rover' or self.use_rover_stream:
            if self.rover_stream_url:
                print(f"📹 Rover stream: {self.rover_stream_url}")
                cap = cv2.VideoCapture(self.rover_stream_url)
            else:
                print("❌ Rover stream not configured")
                return
        elif isinstance(video_path_or_source, int):
            print(f"📹 Camera: {video_path_or_source}")
            cap = cv2.VideoCapture(video_path_or_source)
        else:
            print(f"📹 Video: {video_path_or_source}")
            cap = cv2.VideoCapture(video_path_or_source)
        
        if not cap.isOpened():
            print(f"❌ Cannot open: {video_path_or_source}")
            return
        
        frame_count = 0
        emergency_stop_triggered = False
        last_rotation_state = None
        
        print("\n⚠️  CONTROLS:")
        print("   Q = Emergency stop | R = Resume | ESC = Exit")
        print(f"{'='*80}\n")
        
        while True:
            # Clear buffer for real-time frames
            for _ in range(2):
                cap.grab()
            
            ret, frame = cap.retrieve()
            if not ret:
                ret, frame = cap.read()
                if not ret:
                    print("\n⚠️ Stream ended")
                    break
            
            frame = cv2.resize(frame, (640, 480))
            frame_count += 1
            
            # Emergency stop handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                if not emergency_stop_triggered:
                    emergency_stop_triggered = True
                    self.rover_controller.emergency_stop_now()
                    print("\n🚨 EMERGENCY STOP - Q=exit | R=resume")
                else:
                    print("\n⏹️ Exiting...")
                    break
            elif key == ord('r') or key == ord('R'):
                if emergency_stop_triggered:
                    emergency_stop_triggered = False
                    self.rover_controller.reset_emergency_stop()
                    print("\n✅ Resuming...")
            elif key == 27:
                print("\n⏹️ Stopped (ESC)")
                self.rover_controller.emergency_stop_now()
                break
            
            # Emergency stop overlay
            if emergency_stop_triggered:
                emergency_frame = frame.copy()
                cv2.rectangle(emergency_frame, (0, 0), (640, 100), (0, 0, 255), -1)
                cv2.putText(emergency_frame, "EMERGENCY STOP", (100, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                cv2.putText(emergency_frame, "R=resume | Q=exit", (150, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                map_canvas = self.draw_map_with_live_position(frame)
                cv2.imshow("Navigation Map", map_canvas)
                cv2.imshow("Current Frame", emergency_frame)
                continue
            
            # Check rotation state
            current_rotation_state = self.rover_controller.rotation_state
            rotation_in_progress = self.rover_controller.update_rotation_state_machine()
            
            # Clear frame buffer when rotation completes
            if last_rotation_state == 'cooldown' and current_rotation_state is None:
                print("🧹 Clearing frame buffer...")
                for _ in range(10):
                    cap.grab()
                self.current_pose = None
                print("✅ Fresh localization starting")
            
            last_rotation_state = current_rotation_state
            
            # Rotation overlay
            if rotation_in_progress:
                rotation_frame = frame.copy()
                cv2.rectangle(rotation_frame, (0, 0), (640, 100), (0, 165, 255), -1)
                
                state_text = {
                    'pre_stop': "STOPPING",
                    'rotating': "ROTATING 90°",
                    'post_stop': "STABILIZING",
                    'cooldown': "CLEARING"
                }.get(current_rotation_state, "ROTATION")
                
                cv2.putText(rotation_frame, state_text, (200, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                cv2.putText(rotation_frame, "Navigation Paused", (180, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                map_canvas = self.draw_map_with_live_position(frame)
                cv2.imshow("Navigation Map", map_canvas)
                cv2.imshow("Current Frame", rotation_frame)
                continue
            
            # Localization
            start_time = time.time()
            matched_kf_id, confidence, inliers, estimated_pose = self.localize_frame(frame)
            elapsed = time.time() - start_time
            
            # Update pose
            if matched_kf_id is not None and estimated_pose is not None:
                matched_kf = self.all_keyframes[matched_kf_id]
                old_pose = self.current_pose
                
                self.current_pose = matched_kf.pose.copy()
                self.current_kf_id = matched_kf_id
                
                if old_pose is not None:
                    dx = self.current_pose[0] - old_pose[0]
                    dy = self.current_pose[1] - old_pose[1]
                    distance = np.sqrt(dx**2 + dy**2)
                    
                    if distance < 0.5:
                        self.total_distance += distance
                        self.trajectory.append((self.current_pose[0], self.current_pose[1]))
                    else:
                        print(f"  ⚠️ Jump: {distance:.2f}m")
                else:
                    self.trajectory.append((self.current_pose[0], self.current_pose[1]))
                
                print(f"[{frame_count:5d}] ✅ KF#{matched_kf_id} | "
                      f"Conf: {confidence*100:.1f}% | Inliers: {inliers} | "
                      f"{elapsed*1000:.1f}ms")
            else:
                print(f"[{frame_count:5d}] ❌ No match")
            
            # Fresh navigation calculation
            nav_direction = self.get_navigation_direction()
            
            if matched_kf_id is None:
                nav_direction = {
                    'direction': 'forward',
                    'angle': 0,
                    'distance': 0.3,
                    'target_frame': None,
                    'status': 'searching',
                    'command': 'forward'
                }
            
            self.print_navigation_command(nav_direction)
            
            # Execute command
            if self.auto_control:
                self.rover_controller.execute_navigation_command(nav_direction, matched_kf_id)
            
            # Check destination
            if nav_direction.get('status') == 'destination_reached':
                if self.auto_control:
                    print("\n🛑 Destination reached - STOP")
                    self.rover_controller.send_command('stop')
                frame_with_nav = self.draw_navigation_overlay(frame, nav_direction)
                map_canvas = self.draw_map_with_live_position(frame)
                cv2.imshow("Navigation Map", map_canvas)
                cv2.imshow("Current Frame", frame_with_nav)
                cv2.waitKey(2000)
                break
            
            # Visualization
            frame_with_nav = self.draw_navigation_overlay(frame, nav_direction)
            map_canvas = self.draw_map_with_live_position(frame)
            cv2.imshow("Navigation Map", map_canvas)
            cv2.imshow("Current Frame", frame_with_nav)
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        if self.auto_control:
            print("\n🛑 Final stop...")
            self.rover_controller.send_command('stop')
        
        print(f"\n{'='*80}")
        print("🎯 COMPLETE")
        print(f"   Distance: {self.total_distance:.3f}m | Frames: {frame_count}")
        print(f"{'='*80}\n")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*80)
    print("🚀 LIVE ROVER NAVIGATION")
    print("="*80 + "\n")
    
    print("🎮 Control Mode:")
    print("  1. Simulation")
    print("  2. Automatic Rover Control")
    mode = input("Select (1/2, default 1): ").strip()
    
    enable_rover = (mode == '2')
    
    if enable_rover:
        print("\n🤖 AUTOMATIC CONTROL ENABLED")
        rover_ip = input("Rover IP (default 10.226.57.127): ").strip()
        rover_ip = rover_ip if rover_ip else "10.226.57.127"
        rover_port = input("Port (default 8080): ").strip()
        rover_port = int(rover_port) if rover_port else 8080
    else:
        print("\n🔒 SIMULATION MODE")
        rover_ip = "10.226.57.127"
        rover_port = 8080
    
    print("\n📹 Video Source:")
    print("  1. Video file")
    print("  2. Camera")
    print("  3. Rover stream")
    source_type = input("Select (1/2/3, default 1): ").strip()
    
    use_rover_stream = False
    video_source = None
    
    if source_type == '2':
        camera_idx = input("Camera index (default 0): ").strip()
        video_source = int(camera_idx) if camera_idx else 0
        print(f"✅ Camera: {video_source}")
    elif source_type == '3':
        use_rover_stream = True
        video_source = 'rover'
        print(f"✅ Rover stream: {rover_ip}:{rover_port}")
    else:
        video_source = input("Video file path: ").strip()
        if not os.path.exists(video_source):
            print(f"❌ Not found: {video_source}")
            return
        print(f"✅ Video: {video_source}")
    
    # Initialize system
    nav = LiveNavigationSystem(
        fx=600.0, fy=600.0, cx=320.0, cy=240.0,
        enable_rover_control=enable_rover,
        rover_ip=rover_ip,
        rover_port=rover_port,
        use_rover_stream=use_rover_stream
    )
    
    # Load map
    if not nav.load_map("keyframes_storage"):
        print("❌ Failed to load map")
        return
    
    # Select destination
    all_kf_ids = sorted(nav.all_keyframes.keys())
    print(f"\n📍 Keyframes: {len(all_kf_ids)} (KF#{all_kf_ids[0]} to KF#{all_kf_ids[-1]})")
    
    dest_input = input(f"Destination KF# (default {all_kf_ids[-1]}): ").strip()
    
    if dest_input:
        try:
            destination_kf = int(dest_input)
            if destination_kf in all_kf_ids:
                nav.destination_kf_id = destination_kf
                print(f"✅ Destination: KF#{destination_kf}")
            else:
                print(f"⚠️ Invalid, using KF#{all_kf_ids[-1]}")
                nav.destination_kf_id = all_kf_ids[-1]
        except ValueError:
            print(f"⚠️ Invalid, using KF#{all_kf_ids[-1]}")
            nav.destination_kf_id = all_kf_ids[-1]
    else:
        nav.destination_kf_id = all_kf_ids[-1]
        print(f"✅ Destination: KF#{all_kf_ids[-1]}")
    
    print("\n" + "="*80)
    print("⚠️  Q=Emergency Stop | R=Resume | ESC=Exit")
    if enable_rover:
        print("⚠️  ROVER WILL MOVE AUTOMATICALLY")
    else:
        print("ℹ️  Simulation mode")
    print("="*80 + "\n")
    
    nav.run_live_navigation(video_source, process_every_n=5)


if __name__ == "__main__":
    main()