"""
Configuration Management
"""

import os
import json


class Config:
    """Application configuration"""

    def __init__(self):
        # Server settings
        self.HOST = os.getenv('MEDIBOT_HOST', '0.0.0.0')
        self.PORT = int(os.getenv('MEDIBOT_PORT', 5000))

        # Rover settings
        self.DEFAULT_ROVER_IP = os.getenv('ROVER_IP', '192.168.1.100')
        self.DEFAULT_ROVER_PORT = int(os.getenv('ROVER_PORT', 8080))

        # Paths
        self.BASE_DIR = '/home/user/Medibot'
        self.KEYFRAMES_DIR = os.path.join(self.BASE_DIR, 'keyframes_storage')
        self.DATA_DIR = os.path.join(self.BASE_DIR, 'data')
        self.PATIENTS_FILE = os.path.join(self.DATA_DIR, 'patients', 'patients.json')

        # Camera settings
        self.CAMERA_WIDTH = 640
        self.CAMERA_HEIGHT = 480
        self.CAMERA_FPS = 40

        # Navigation settings
        self.MAX_KEYFRAMES = 26
        self.LOCALIZATION_THRESHOLD = 0.1
        self.MOVEMENT_TIMEOUT = 30  # seconds

        # LLM settings (for future use)
        self.LLM_ENABLED = False
        self.LLM_MODEL = "gpt-3.5-turbo"

    def to_dict(self):
        """Convert config to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


def load_config(config_file: str = None) -> Config:
    """Load configuration from file or environment"""
    config = Config()

    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            custom_config = json.load(f)

        for key, value in custom_config.items():
            if hasattr(config, key):
                setattr(config, key, value)

    return config
