/*
 * Configuration
 */

const CONFIG = {
    // API Configuration
    API_BASE_URL: window.location.origin,
    API_ENDPOINTS: {
        // Rover
        ROVER_CONNECT: '/api/rover/connect',
        ROVER_DISCONNECT: '/api/rover/disconnect',
        ROVER_STATUS: '/api/rover/status',
        ROVER_MOVE: '/api/rover/move',
        ROVER_STOP: '/api/rover/stop',

        // Camera
        CAMERA_FEED: '/api/camera/feed',
        CAMERA_SNAPSHOT: '/api/camera/snapshot',

        // Map
        MAP_KEYFRAMES: '/api/map/keyframes',
        MAP_FEATURES: '/api/map/features',

        // Navigation
        NAV_GOTO: '/api/navigation/goto',

        // Patients
        PATIENTS: '/api/patients',
        PATIENT_DELIVER: (id) => `/api/patients/${id}/deliver`
    },

    // WebSocket Configuration
    SOCKET_URL: window.location.origin,

    // Default Settings
    DEFAULT_ROVER_IP: '192.168.1.100',
    ROVER_PORT: 8080,

    // Camera Settings
    CAMERA_UPDATE_INTERVAL: 100, // ms

    // Map Settings
    MAP: {
        CANVAS_WIDTH: 800,
        CANVAS_HEIGHT: 800,
        BACKGROUND_COLOR: '#FFFFFF',
        FEATURE_COLOR: '#000000',
        KEYFRAME_COLOR: '#4CAF50',
        ROVER_COLOR: '#2196F3',
        TARGET_COLOR: '#FF5722',
        FEATURE_SIZE: 1,
        KEYFRAME_SIZE: 8,
        ROVER_SIZE: 10,
        GRID_COLOR: '#EEEEEE',
        TEXT_COLOR: '#666666'
    },

    // Update Intervals
    STATUS_UPDATE_INTERVAL: 1000, // ms
    MAP_UPDATE_INTERVAL: 500, // ms

    // UI Settings
    ANIMATION_DURATION: 300, // ms
    TOAST_DURATION: 3000 // ms
};

// Freeze config to prevent modifications
Object.freeze(CONFIG);
