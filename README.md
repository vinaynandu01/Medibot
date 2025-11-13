# 🤖 MEDIBOT - Medication Delivery Rover System

> An intelligent autonomous indoor navigation system for medication delivery in healthcare facilities

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Configuration](#configuration)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🌟 Overview

The **Medication Delivery Rover** is an advanced autonomous ground robot designed to transport and deliver medications within hospitals or elderly care centers. It utilizes state-of-the-art Computer Vision for navigation and patient identification, combined with Generative AI for patient interaction and automated logging.

### Problem Statement

In healthcare facilities, medication delivery is:
- **Time-consuming** for nursing staff
- **Prone to human error** in timing and dosage
- **Resource-intensive** requiring constant attention
- **Inconsistent** during shift changes and high-demand periods

**Medibot** addresses these challenges through autonomous delivery, reducing workload, minimizing errors, and improving reliability.

## ✨ Features

### 🗺️ Navigation & Mapping
- **Visual SLAM** using ORB features and monocular depth estimation
- **Real-time localization** against pre-built keyframe maps
- **26-keyframe navigation system** with path planning
- **Live map visualization** with white background and black feature points
- **Interactive map selection** for goal-point navigation

### 📹 Computer Vision
- **Live camera feed** streaming at 640×480 @ 40fps
- **MiDaS depth estimation** for 3D feature triangulation
- **ORB feature detection** with CLAHE enhancement
- **PnP RANSAC** pose estimation
- **Face recognition ready** (integration pending)

### 🏥 Patient Management
- **Complete patient database** with medication schedules
- **Keyframe-based location assignment** for each patient
- **Delivery tracking** and history
- **Schedule management** for medication timing
- **Interactive patient cards** with edit/delete/deliver actions

### 🎮 Rover Control
- **Web-based control interface** accessible from any device
- **Joystick controls** (forward, backward, left, right, stop)
- **Keyboard shortcuts** (WASD/Arrow keys + Space for stop)
- **Queue-based command system** for sequential execution
- **Emergency stop** functionality

### 📊 Real-Time Monitoring
- **Live rover position** visualization on map
- **System status dashboard** with statistics
- **Activity logging** with timestamps
- **Battery monitoring** with visual indicators
- **WebSocket updates** for instant feedback

### 💻 Modern Web Interface
- **Responsive CSS Grid layout** (2025 design patterns)
- **Auto-fit grid** with minmax() for adaptive design
- **Mobile-friendly** interface with touch support
- **Dark mode ready** architecture
- **96%+ browser compatibility**

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    WEB INTERFACE (Frontend)                 │
│  HTML5 + Modern CSS Grid + Vanilla JavaScript + Socket.IO  │
└─────────────────────────────────────────────────────────────┘
                              ↕ HTTP/WebSocket
┌─────────────────────────────────────────────────────────────┐
│              BACKEND API (Flask + Flask-SocketIO)           │
│   • Rover Control API     • Patient Management API         │
│   • Map & Navigation API  • Real-time Communication        │
└─────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────┐
│                   ROVER HARDWARE LAYER                      │
│   • CSI Camera (GStreamer)    • Motor Controller (Serial)  │
│   • Movement Calibration      • Command Queue               │
└─────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────┐
│                 NAVIGATION & SLAM LAYER                     │
│   • MiDaS Depth Estimation   • ORB Feature Extraction       │
│   • Keyframe Localization    • Path Planning               │
│   • 3D Feature Triangulation • Pose Estimation (PnP)        │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Node.js 16+ (for Socket.IO CDN, or use local)
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Rover hardware (optional for testing)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/medibot.git
cd medibot
```

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Verify Directory Structure

```bash
ls -la
# Should see: backend/, frontend/, rover/, data/, docs/
```

### Step 4: Configure Settings (Optional)

```bash
# Edit backend configuration
nano backend/utils/config.py

# Or set environment variables
export MEDIBOT_PORT=5000
export ROVER_IP=192.168.1.100
```

## 🎯 Usage

### Starting the System

#### 1. Start Backend Server

```bash
python backend/app.py
```

The server will start on `http://localhost:5000`

#### 2. Open Web Interface

Navigate to `http://localhost:5000` in your browser

#### 3. Connect to Rover

1. Enter rover IP address (default: 192.168.1.100)
2. Click **Connect**
3. Wait for confirmation

#### 4. Start Navigation

**Option A: Manual Control**
- Use on-screen joystick buttons
- Or use keyboard: `W/↑` (forward), `S/↓` (backward), `A/←` (left), `D/→` (right), `Space` (stop)

**Option B: Patient Delivery**
1. Go to **Patients** panel
2. Click **Deliver** on a patient card
3. Rover automatically navigates to assigned keyframe

**Option C: Map Selection**
1. Click on a keyframe in the **Navigation Map**
2. Confirm navigation
3. Watch rover path in real-time

### Running Navigation System Standalone

```bash
# Live navigation with rover camera
python rover/navigation/navig.py --source rover --rover-ip 192.168.1.100

# Navigation with video file
python rover/navigation/navig.py --source video --video-path path/to/video.mp4

# Localization only mode
python rover/navigation/local.py
```

### Building a New Map

```bash
# Record video while moving rover
python rover/slam/map_cap.py --rover-ip 192.168.1.100 --output map_video.mp4

# Build keyframe map
python rover/slam/tuned.py --video-path map_video.mp4 --output-dir data/keyframes
```

## 📁 Project Structure

```
Medibot/
├── backend/                    # Flask backend server
│   ├── app.py                 # Main application entry
│   ├── api/                   # API modules
│   │   ├── rover_api.py       # Rover control interface
│   │   ├── patient_api.py     # Patient management
│   │   └── map_api.py         # Map and keyframe data
│   ├── models/                # Data models
│   │   └── patient.py         # Patient model & manager
│   └── utils/                 # Utilities
│       └── config.py          # Configuration management
│
├── frontend/                  # Web interface
│   ├── index.html            # Main HTML page
│   ├── css/                  # Stylesheets
│   │   ├── styles.css        # Base styles & variables
│   │   ├── dashboard.css     # Dashboard grid layout
│   │   ├── map.css           # Map visualization
│   │   └── patients.css      # Patient panel styles
│   └── js/                   # JavaScript modules
│       ├── config.js         # Configuration
│       ├── api.js            # API client
│       ├── map.js            # Map visualization
│       ├── patients.js       # Patient management
│       ├── dashboard.js      # Dashboard controller
│       └── main.js           # Application entry point
│
├── rover/                     # Rover code modules
│   ├── control/              # Hardware control
│   │   └── server.py         # Rover control server
│   ├── navigation/           # Navigation system
│   │   ├── navig.py         # Live navigation
│   │   └── local.py         # Localization only
│   └── slam/                 # SLAM & mapping
│       ├── tuned.py         # Map building
│       └── map_cap.py       # Video recording
│
├── data/                      # Data storage
│   ├── keyframes/            # Keyframe map data
│   │   ├── keyframes_index.json
│   │   ├── direction_map.json
│   │   └── keyframe_*.pkl
│   ├── patients/             # Patient database
│   │   └── patients.json
│   └── config/               # Configuration files
│       └── depth_scale_factor.npy
│
├── docs/                      # Documentation
├── tests/                     # Unit tests
├── requirements.txt           # Python dependencies
├── README.md                 # This file
└── LICENSE                   # License file
```

## 📡 API Documentation

### Rover Endpoints

#### POST `/api/rover/connect`
Connect to rover hardware
```json
{
  "ip_address": "192.168.1.100"
}
```

#### POST `/api/rover/move`
Send movement command
```json
{
  "direction": "forward" // or "backward", "left", "right", "stop"
}
```

#### GET `/api/rover/status`
Get current rover status

### Patient Endpoints

#### GET `/api/patients`
Get all patients

#### POST `/api/patients`
Add new patient
```json
{
  "name": "John Doe",
  "room_number": "A101",
  "medication": "Aspirin 100mg",
  "keyframe_id": 5,
  "schedule": "3 times daily",
  "notes": "Take with food"
}
```

#### POST `/api/patients/{id}/deliver`
Start delivery to patient

### Map Endpoints

#### GET `/api/map/keyframes`
Get all keyframes

#### GET `/api/map/features`
Get map feature points

### WebSocket Events

- `rover_connected` - Rover connection established
- `rover_state` - Real-time rover state update
- `rover_moved` - Movement command executed
- `navigation_started` - Navigation to keyframe started
- `delivery_started` - Delivery to patient started

## ⚙️ Configuration

### Backend Configuration

Edit `backend/utils/config.py`:

```python
# Server settings
HOST = '0.0.0.0'
PORT = 5000

# Rover settings
DEFAULT_ROVER_IP = '192.168.1.100'
DEFAULT_ROVER_PORT = 8080

# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 40
```

### Frontend Configuration

Edit `frontend/js/config.js`:

```javascript
const CONFIG = {
    API_BASE_URL: window.location.origin,
    DEFAULT_ROVER_IP: '192.168.1.100',
    CAMERA_UPDATE_INTERVAL: 100,
    // ... more settings
};
```

### Rover Hardware Configuration

Edit `rover/control/server.py`:

```python
# Movement calibration
FORWARD_SPEED = 50      # Motor speed (0-100)
TURN_SPEED = 40
FORWARD_DURATION = 1.2  # seconds for 0.3m
TURN_DURATION = 3.5     # seconds for 90°
```

## 🛠️ Development

### Running Tests

```bash
# Backend tests
python -m pytest tests/

# Frontend tests (if using Jest)
npm test
```

### Code Style

```bash
# Python code formatting
black backend/ rover/

# Python linting
pylint backend/ rover/

# JavaScript linting
eslint frontend/js/
```

### Building Documentation

```bash
# Generate API docs
python -m pdoc backend --html --output-dir docs/api

# Generate coverage report
pytest --cov=backend --cov-report=html
```

## 🐛 Troubleshooting

### Rover Not Connecting

1. Check IP address is correct
2. Verify rover is powered on
3. Check network connectivity: `ping 192.168.1.100`
4. Verify rover server is running on port 8080

### Camera Feed Not Showing

1. Check rover is connected
2. Verify camera permissions in browser
3. Check rover camera is working: visit `http://ROVER_IP:8080/video_feed`
4. Clear browser cache

### Map Not Loading

1. Check keyframes exist in `data/keyframes/`
2. Verify `keyframes_index.json` is valid JSON
3. Check browser console for errors
4. Try refreshing the page

### Navigation Not Working

1. Ensure rover is connected
2. Check target keyframe is selected on map
3. Verify direction_map.json exists
4. Check activity log for error messages

## 🎓 Learning Outcomes

This project demonstrates:

1. ✅ **Indoor robot navigation** using CV-based SLAM
2. ✅ **Computer vision** integration (ORB features, depth estimation)
3. ✅ **Real-time decision-making** with sensor fusion
4. ✅ **Full-stack robotics system** (perception → decision → action)
5. ✅ **Modern web development** (responsive CSS Grid, WebSockets)
6. ✅ **RESTful API design** and real-time communication
7. ✅ **Database management** for patient records
8. ✅ **User interface design** for monitoring and control
9. ✅ **Healthcare workflow** automation
10. ✅ **Real-world prototyping** with commercial potential

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👏 Acknowledgments

- **MiDaS** depth estimation by Intel ISL
- **OpenCV** for computer vision capabilities
- **Flask** and **Socket.IO** for web framework
- Modern CSS Grid patterns from the web development community

## 📧 Contact

Project Link: [https://github.com/yourusername/medibot](https://github.com/yourusername/medibot)

---

**Made with ❤️ for improving healthcare delivery**
