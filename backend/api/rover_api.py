"""
Rover API Module
Handles communication with the rover hardware
"""

import requests
import time
from typing import Optional


class RoverAPI:
    """Interface for rover control and communication"""

    def __init__(self, ip_address: str, port: int = 8080):
        self.ip_address = ip_address
        self.port = port
        self.base_url = f"http://{ip_address}:{port}"

    def move(self, direction: str, duration: Optional[float] = None) -> dict:
        """
        Send movement command to rover

        Args:
            direction: 'forward', 'backward', 'left', 'right', 'stop'
            duration: Optional duration in seconds

        Returns:
            Response from rover
        """
        try:
            response = requests.post(
                f"{self.base_url}/move",
                json={'direction': direction},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"Failed to move rover: {str(e)}")

    def stop(self) -> dict:
        """Emergency stop"""
        try:
            # Clear command queue
            requests.post(f"{self.base_url}/queue/clear", timeout=5)

            # Send stop command
            response = requests.post(
                f"{self.base_url}/move",
                json={'direction': 'stop'},
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"Failed to stop rover: {str(e)}")

    def get_queue_status(self) -> dict:
        """Get current command queue status"""
        try:
            response = requests.get(f"{self.base_url}/queue/status", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"Failed to get queue status: {str(e)}")

    def clear_queue(self) -> dict:
        """Clear command queue"""
        try:
            response = requests.post(f"{self.base_url}/queue/clear", timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise Exception(f"Failed to clear queue: {str(e)}")

    def get_camera_url(self) -> str:
        """Get camera feed URL"""
        return f"{self.base_url}/video_feed"

    def get_snapshot_url(self) -> str:
        """Get snapshot URL"""
        return f"{self.base_url}/frame"

    def navigate_to_keyframe(self, keyframe_id: int, patient_id: Optional[str] = None):
        """
        Navigate to specific keyframe
        This integrates with the navigation system (navig.py)

        Args:
            keyframe_id: Target keyframe ID
            patient_id: Optional patient ID for delivery tracking
        """
        # This will integrate with the navigation system
        # For now, we'll implement basic navigation logic

        print(f"Navigating to keyframe {keyframe_id}")

        if patient_id:
            print(f"Delivering medication to patient {patient_id}")

        # Load direction map
        import json
        import os
        try:
            # Try new location first, fallback to old location
            direction_map_path = None
            if os.path.exists('/home/user/Medibot/data/keyframes/direction_map.json'):
                direction_map_path = '/home/user/Medibot/data/keyframes/direction_map.json'
            elif os.path.exists('/home/user/Medibot/keyframes_storage/direction_map.json'):
                direction_map_path = '/home/user/Medibot/keyframes_storage/direction_map.json'
            else:
                raise FileNotFoundError("direction_map.json not found")

            with open(direction_map_path, 'r') as f:
                direction_map = json.load(f)

            # Simple navigation: follow direction map
            current_kf = 0  # Start from keyframe 0

            while current_kf != keyframe_id:
                kf_key = f"KF#{current_kf}"

                if kf_key not in direction_map:
                    print(f"No direction found for {kf_key}")
                    break

                direction_info = direction_map[kf_key]
                direction = direction_info['direction']

                print(f"At {kf_key}, moving {direction}")

                # Execute movement based on direction
                if 'rotate' in direction:
                    self.move('left' if 'left' in direction else 'right')
                    time.sleep(3.5)  # 90-degree turn
                elif direction == 'forward':
                    self.move('forward')
                    time.sleep(1.2)  # 0.3m forward
                elif 'forward_left' in direction:
                    self.move('left')
                    time.sleep(1.75)  # 45-degree turn
                    self.move('forward')
                    time.sleep(1.2)
                elif 'forward_right' in direction:
                    self.move('right')
                    time.sleep(1.75)  # 45-degree turn
                    self.move('forward')
                    time.sleep(1.2)

                self.move('stop')
                time.sleep(0.5)

                # Move to next keyframe
                current_kf += 1

                if current_kf > 25:  # Max keyframe
                    break

            print(f"Reached keyframe {keyframe_id}")

            if patient_id:
                print(f"Medication delivered to patient {patient_id}")

        except Exception as e:
            print(f"Navigation error: {e}")
            raise

    def test_connection(self) -> bool:
        """Test if rover is reachable"""
        try:
            response = requests.get(f"{self.base_url}/queue/status", timeout=5)
            return response.status_code == 200
        except:
            return False
