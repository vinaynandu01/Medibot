"""
Map API Module
Handles map data and keyframe information
"""

import json
import pickle
import os
import numpy as np
from typing import Dict, List, Optional


class MapAPI:
    """Interface for map and keyframe data"""

    def __init__(self, keyframes_dir: str = None):
        # Try new location first, fallback to old location
        if keyframes_dir is None:
            # Get the base directory (Medibot root)
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            base_dir = os.path.dirname(base_dir)  # Go up one more level from backend
            keyframes_path = os.path.join(base_dir, 'data', 'keyframes')
            
            if os.path.exists(os.path.join(keyframes_path, 'keyframes_index.json')):
                keyframes_dir = keyframes_path
            else:
                keyframes_dir = os.path.join(base_dir, 'keyframes_storage')

        self.keyframes_dir = keyframes_dir
        self.keyframes_index = None
        self.direction_map = None
        self.load_map_data()

    def load_map_data(self):
        """Load keyframes index and direction map"""
        try:
            # Load keyframes index
            index_path = os.path.join(self.keyframes_dir, 'keyframes_index.json')
            with open(index_path, 'r') as f:
                self.keyframes_index = json.load(f)

            # Load direction map
            direction_path = os.path.join(self.keyframes_dir, 'direction_map.json')
            with open(direction_path, 'r') as f:
                self.direction_map = json.load(f)

            # Count keyframes properly
            num_keyframes = len(self.keyframes_index.get('keyframes', {}))
            print(f"Loaded {num_keyframes} keyframes")

        except Exception as e:
            print(f"Error loading map data: {e}")
            self.keyframes_index = {}
            self.direction_map = {}

    def get_keyframes(self) -> List[Dict]:
        """
        Get all keyframes with their positions

        Returns:
            List of keyframes with id, pose, and feature count
        """
        if not self.keyframes_index:
            return []

        keyframes = []
        keyframes_dict = self.keyframes_index.get('keyframes', {})
        
        for idx_str, kf in keyframes_dict.items():
            keyframes.append({
                'id': int(idx_str),
                'pose': kf.get('pose', [0, 0, 0]),
                'num_features': kf.get('num_features', 0),
                'timestamp': kf.get('timestamp', '')
            })

        return keyframes

    def get_keyframe_by_id(self, keyframe_id: int) -> Optional[Dict]:
        """Get specific keyframe by ID"""
        keyframes_dict = self.keyframes_index.get('keyframes', {})
        kf = keyframes_dict.get(str(keyframe_id))
        
        if kf:
            return {
                'id': keyframe_id,
                'pose': kf.get('pose', [0, 0, 0]),
                'num_features': kf.get('num_features', 0),
                'timestamp': kf.get('timestamp', '')
            }
        return None

    def get_features(self) -> List[Dict]:
        """
        Get all feature points from all keyframes for map visualization

        Returns:
            List of feature points with x, y coordinates
        """
        features = []

        try:
            keyframes_dict = self.keyframes_index.get('keyframes', {})
            
            for idx_str, kf_info in keyframes_dict.items():
                idx = int(idx_str)
                # Get filename from keyframes_index
                pkl_file = kf_info.get('filename', f'keyframe_{idx:06d}.pkl')
                
                # Handle both absolute and relative paths
                # If path contains subdirectory reference, extract just the filename
                pkl_filename = os.path.basename(pkl_file)
                pkl_path = os.path.join(self.keyframes_dir, pkl_filename)

                if os.path.exists(pkl_path):
                    with open(pkl_path, 'rb') as f:
                        kf_data = pickle.load(f)

                    # Get 3D features
                    if hasattr(kf_data, 'features_3d') and kf_data.features_3d:
                        for feat in kf_data.features_3d:
                            # Extract x, y from 3D feature
                            features.append({
                                'x': float(feat[0]),
                                'y': float(feat[1]),
                                'keyframe_id': idx
                            })

        except Exception as e:
            print(f"Error loading features: {e}")

        return features

    def get_direction_map(self) -> Dict:
        """Get the direction map for navigation"""
        return self.direction_map

    def get_map_bounds(self) -> Dict:
        """
        Get map boundaries for visualization

        Returns:
            Dict with min_x, max_x, min_y, max_y
        """
        if not self.keyframes_index:
            return {'min_x': 0, 'max_x': 1, 'min_y': 0, 'max_y': 1}

        x_coords = [kf['pose'][0] for kf in self.keyframes_index if 'pose' in kf]
        y_coords = [kf['pose'][1] for kf in self.keyframes_index if 'pose' in kf]

        if not x_coords or not y_coords:
            return {'min_x': 0, 'max_x': 1, 'min_y': 0, 'max_y': 1}

        # Add padding
        padding = 0.5
        return {
            'min_x': min(x_coords) - padding,
            'max_x': max(x_coords) + padding,
            'min_y': min(y_coords) - padding,
            'max_y': max(y_coords) + padding
        }

    def get_path_between_keyframes(self, start_id: int, end_id: int) -> List[int]:
        """
        Get path between two keyframes

        Returns:
            List of keyframe IDs forming the path
        """
        # Simple path: just sequential keyframes
        # In a real implementation, this would use graph search
        if start_id <= end_id:
            return list(range(start_id, end_id + 1))
        else:
            return list(range(start_id, end_id - 1, -1))
