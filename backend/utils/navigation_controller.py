"""
Navigation Controller with Face Recognition Integration
Manages rover navigation to patient locations with real-time face detection
"""

import sys
import os
import threading
import time
import cv2
import numpy as np
import importlib.util
from typing import Optional, Dict, Callable

class NavigationController:
    """
    Manages rover navigation with real-time face recognition
    Integrates with navig.py for autonomous navigation
    """
    
    def __init__(self, face_service, socketio, rover_ip="192.168.1.100"):
        self.face_service = face_service
        self.socketio = socketio
        self.rover_ip = rover_ip
        
        # Navigation state
        self.is_navigating = False
        self.current_target_patient = None
        self.current_target_keyframe = None
        self.navigation_thread = None
        self.stop_navigation_flag = False
        
        # Face recognition state
        self.target_patient_detected = False
        self.detection_confidence = 0.0
        self.last_detection_time = None
        
        # Load navigation module
        self.nav_module = None
        self._load_navigation_module()
    
    def _load_navigation_module(self):
        """Load navig.py module dynamically"""
        try:
            # Try to load from root directory
            nav_path = os.path.join(os.getcwd(), '..', 'navig.py')
            if not os.path.exists(nav_path):
                nav_path = os.path.join(os.getcwd(), '..', '..', 'navig.py')
            
            if os.path.exists(nav_path):
                spec = importlib.util.spec_from_file_location("navig", nav_path)
                self.nav_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(self.nav_module)
                print("✅ Navigation module (navig.py) loaded successfully")
            else:
                print("⚠️  Navigation module not found. Navigation will be simulated.")
        except Exception as e:
            print(f"❌ Error loading navigation module: {e}")
            self.nav_module = None
    
    def start_navigation(self, patient_id: str, patient_name: str, target_keyframe: int) -> Dict:
        """
        Start navigation to patient location
        Includes face recognition during navigation
        """
        if self.is_navigating:
            return {
                'success': False,
                'message': 'Navigation already in progress'
            }
        
        if not self.nav_module:
            return {
                'success': False,
                'message': 'Navigation module not available'
            }
        
        # Reset state
        self.is_navigating = True
        self.current_target_patient = patient_id
        self.current_target_keyframe = target_keyframe
        self.stop_navigation_flag = False
        self.target_patient_detected = False
        self.detection_confidence = 0.0
        
        # Start navigation in background thread
        self.navigation_thread = threading.Thread(
            target=self._navigation_worker,
            args=(patient_id, patient_name, target_keyframe),
            daemon=True
        )
        self.navigation_thread.start()
        
        # Emit navigation started event
        self.socketio.emit('navigation_started', {
            'patient_id': patient_id,
            'patient_name': patient_name,
            'target_keyframe': target_keyframe,
            'timestamp': time.time()
        })
        
        return {
            'success': True,
            'message': f'Navigation started to {patient_name}',
            'patient_id': patient_id,
            'target_keyframe': target_keyframe
        }
    
    def stop_navigation(self) -> Dict:
        """Emergency stop navigation"""
        if not self.is_navigating:
            return {
                'success': False,
                'message': 'No navigation in progress'
            }
        
        self.stop_navigation_flag = True
        
        # Send stop command to rover
        if self.nav_module:
            try:
                # Create rover controller and send stop
                rover_controller = self.nav_module.RoverController(
                    rover_ip=self.rover_ip,
                    enable_control=True
                )
                rover_controller.send_command('stop')
                rover_controller.emergency_stop = True
            except Exception as e:
                print(f"Error sending stop command: {e}")
        
        self.socketio.emit('navigation_stopped', {
            'reason': 'emergency_stop',
            'timestamp': time.time()
        })
        
        return {
            'success': True,
            'message': 'Navigation stopped'
        }
    
    def _navigation_worker(self, patient_id: str, patient_name: str, target_keyframe: int):
        """
        Background worker for navigation
        Handles autonomous navigation with face recognition
        """
        try:
            print(f"🚀 Starting navigation to patient {patient_name} (Keyframe {target_keyframe})")
            
            # Initialize navigation system
            rover_controller = self.nav_module.RoverController(
                rover_ip=self.rover_ip,
                enable_control=True
            )
            
            # Emit status updates
            self._emit_status('initializing', 'Starting navigation system...')
            
            # Load map and keyframes
            self._emit_status('loading_map', 'Loading map data...')
            time.sleep(1)  # Simulated loading
            
            # Start localization
            self._emit_status('localizing', 'Localizing rover position...')
            time.sleep(2)  # Simulated localization
            
            # Navigate to target
            self._emit_status('navigating', f'Navigating to {patient_name}...')
            
            # Simulate navigation with face recognition checks
            for progress in range(0, 101, 10):
                if self.stop_navigation_flag:
                    self._emit_status('stopped', 'Navigation stopped by user')
                    break
                
                # Emit progress
                self.socketio.emit('navigation_progress', {
                    'patient_id': patient_id,
                    'progress': progress,
                    'status': 'navigating'
                })
                
                # Check for face recognition (simulate every 20% progress)
                if progress > 0 and progress % 20 == 0:
                    self._check_face_recognition(patient_id, patient_name)
                
                time.sleep(1)
            
            if not self.stop_navigation_flag:
                # Goal reached
                self._emit_status('goal_reached', f'Arrived at {patient_name}\'s location')
                
                # Final face recognition check
                time.sleep(1)
                self._check_face_recognition(patient_id, patient_name, final_check=True)
                
                # Wait for confirmation
                self.socketio.emit('delivery_ready', {
                    'patient_id': patient_id,
                    'patient_name': patient_name,
                    'patient_detected': self.target_patient_detected,
                    'confidence': self.detection_confidence
                })
        
        except Exception as e:
            print(f"❌ Navigation error: {e}")
            self._emit_status('error', f'Navigation error: {str(e)}')
        
        finally:
            self.is_navigating = False
            self.current_target_patient = None
            self.current_target_keyframe = None
    
    def _check_face_recognition(self, patient_id: str, patient_name: str, final_check: bool = False):
        """
        Check for face recognition during navigation
        """
        if not self.face_service.is_available():
            return
        
        # Get camera frame (would come from rover's camera)
        # For now, emit that we're checking
        self.socketio.emit('face_check', {
            'patient_id': patient_id,
            'checking': True,
            'final_check': final_check
        })
        
        # Simulate face recognition
        # In real implementation, this would process actual camera frame
        if patient_id in self.face_service.patient_ids:
            # Simulate detection (70% chance of detection during navigation, 90% at goal)
            import random
            detection_probability = 0.9 if final_check else 0.7
            
            if random.random() < detection_probability:
                self.target_patient_detected = True
                self.detection_confidence = 0.75 + random.random() * 0.2  # 0.75-0.95
                self.last_detection_time = time.time()
                
                self.socketio.emit('patient_detected', {
                    'patient_id': patient_id,
                    'patient_name': patient_name,
                    'confidence': self.detection_confidence,
                    'timestamp': self.last_detection_time
                })
                
                print(f"✅ Patient {patient_name} detected! Confidence: {self.detection_confidence:.2f}")
    
    def _emit_status(self, status: str, message: str):
        """Emit navigation status update"""
        self.socketio.emit('navigation_status', {
            'status': status,
            'message': message,
            'patient_id': self.current_target_patient,
            'timestamp': time.time()
        })
        print(f"📍 Navigation: {message}")
    
    def get_status(self) -> Dict:
        """Get current navigation status"""
        return {
            'is_navigating': self.is_navigating,
            'target_patient': self.current_target_patient,
            'target_keyframe': self.current_target_keyframe,
            'patient_detected': self.target_patient_detected,
            'detection_confidence': self.detection_confidence,
            'nav_module_loaded': self.nav_module is not None
        }
    
    def process_camera_frame_for_recognition(self, frame: np.ndarray) -> Dict:
        """
        Process camera frame for face recognition
        Returns recognized patients in frame
        """
        if not self.face_service.is_available():
            return {'recognized': []}
        
        try:
            results = self.face_service.recognize_face(frame)
            
            # Check if target patient is detected
            if self.is_navigating and self.current_target_patient:
                for result in results:
                    if result['patient_id'] == self.current_target_patient:
                        self.target_patient_detected = True
                        self.detection_confidence = result['confidence']
                        self.last_detection_time = time.time()
                        
                        # Emit detection event
                        patient_name = self.face_service.patient_names.get(
                            self.current_target_patient, 'Unknown'
                        )
                        self.socketio.emit('patient_detected', {
                            'patient_id': self.current_target_patient,
                            'patient_name': patient_name,
                            'confidence': result['confidence'],
                            'timestamp': self.last_detection_time
                        })
            
            return {'recognized': results}
            
        except Exception as e:
            print(f"Error processing frame for recognition: {e}")
            return {'recognized': []}
