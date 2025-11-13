"""
Map API Module
Handles map data and keyframe information
"""

import json
import pickle
import os
import numpy as np
from typing import Dict, List


class MapAPI:
    """Interface for map and keyframe data"""

    def __init__(self, keyframes_dir: str = None):
        # Try new location first, fallback to old location
        if keyframes_dir is None:
            if os.path.exists('/home/user/Medibot/data/keyframes/keyframes_index.json'):
                keyframes_dir = '/home/user/Medibot/data/keyframes'
            else:
                keyframes_dir = '/home/user/Medibot/keyframes_storage'

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

            print(f"Loaded {len(self.keyframes_index)} keyframes")

        except Exception as e:
            print(f"Error loading map data: {e}")
            self.keyframes_index = []
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
        for idx, kf in enumerate(self.keyframes_index):
            keyframes.append({
                'id': idx,
                'pose': kf.get('pose', [0, 0, 0]),
                'num_features': kf.get('num_features', 0),
                'timestamp': kf.get('timestamp', '')
            })

        return keyframes

    def get_keyframe_by_id(self, keyframe_id: int) -> Optional[Dict]:
        """Get specific keyframe by ID"""
        if 0 <= keyframe_id < len(self.keyframes_index):
            kf = self.keyframes_index[keyframe_id]
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
            for idx, kf_info in enumerate(self.keyframes_index):
                # Load pickled keyframe
                pkl_file = kf_info.get('file', f'keyframe_{idx:06d}.pkl')
                pkl_path = os.path.join(self.keyframes_dir, pkl_file)

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
