"""
Medibot Backend API Server
Main Flask application with REST API and WebSocket support
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import json
import os
import time
import threading
from datetime import datetime
import requests
import cv2
import numpy as np

# Import our modules
from api.rover_api import RoverAPI
from api.patient_api import PatientAPI
from api.map_api import MapAPI
from models.patient import PatientManager
from utils.config import Config

app = Flask(__name__, static_folder='../frontend', static_url_path='')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuration
config = Config()
patient_manager = PatientManager()

# Initialize API modules
rover_api = None  # Will be initialized when rover is connected
patient_api = PatientAPI(patient_manager)
map_api = MapAPI()

# Global state
rover_state = {
    'connected': False,
    'ip_address': None,
    'position': {'x': 0, 'y': 0, 'yaw': 0},
    'current_keyframe': None,
    'status': 'idle',  # idle, navigating, delivering, charging
    'battery': 100,
    'last_update': None
}

# ==================== STATIC FILE SERVING ====================

@app.route('/')
def index():
    """Serve the main frontend page"""
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory(app.static_folder, path)

# ==================== ROVER CONNECTION ====================

@app.route('/api/rover/connect', methods=['POST'])
def connect_rover():
    """Connect to the rover with given IP address"""
    global rover_api, rover_state

    data = request.json
    ip_address = data.get('ip_address', '192.168.1.100')

    try:
        # Test connection to rover
        response = requests.get(f'http://{ip_address}:8080/queue/status', timeout=5)

        if response.status_code == 200:
            rover_state['connected'] = True
            rover_state['ip_address'] = ip_address
            rover_state['last_update'] = datetime.now().isoformat()

            # Initialize rover API
            rover_api = RoverAPI(ip_address)

            # Start background thread for rover status updates
            threading.Thread(target=update_rover_status, daemon=True).start()

            # Emit connection status via WebSocket
            socketio.emit('rover_connected', {'ip_address': ip_address})

            return jsonify({
                'success': True,
                'message': f'Connected to rover at {ip_address}',
                'rover_state': rover_state
            })
        else:
            return jsonify({'success': False, 'message': 'Rover not responding'}), 400

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/rover/disconnect', methods=['POST'])
def disconnect_rover():
    """Disconnect from the rover"""
    global rover_api, rover_state

    rover_state['connected'] = False
    rover_state['ip_address'] = None
    rover_api = None

    socketio.emit('rover_disconnected', {})

    return jsonify({'success': True, 'message': 'Disconnected from rover'})

@app.route('/api/rover/status', methods=['GET'])
def get_rover_status():
    """Get current rover status"""
    return jsonify(rover_state)

# ==================== ROVER CONTROL ====================

@app.route('/api/rover/move', methods=['POST'])
def move_rover():
    """Send movement command to rover"""
    if not rover_state['connected'] or not rover_api:
        return jsonify({'success': False, 'message': 'Rover not connected'}), 400

    data = request.json
    direction = data.get('direction')

    try:
        result = rover_api.move(direction)

        # Emit movement update via WebSocket
        socketio.emit('rover_moved', {'direction': direction, 'timestamp': datetime.now().isoformat()})

        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/rover/stop', methods=['POST'])
def stop_rover():
    """Emergency stop"""
    if not rover_state['connected'] or not rover_api:
        return jsonify({'success': False, 'message': 'Rover not connected'}), 400

    try:
        result = rover_api.stop()
        socketio.emit('rover_stopped', {'timestamp': datetime.now().isoformat()})
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# ==================== CAMERA FEED ====================

@app.route('/api/camera/feed')
def camera_feed():
    """Proxy camera feed from rover"""
    if not rover_state['connected']:
        return jsonify({'error': 'Rover not connected'}), 400

    # Proxy the video feed from rover
    rover_url = f"http://{rover_state['ip_address']}:8080/video_feed"

    def generate():
        try:
            response = requests.get(rover_url, stream=True, timeout=10)
            for chunk in response.iter_content(chunk_size=1024):
                yield chunk
        except Exception as e:
            print(f"Error streaming camera: {e}")

    return app.response_class(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/camera/snapshot', methods=['GET'])
def camera_snapshot():
    """Get a single frame snapshot"""
    if not rover_state['connected']:
        return jsonify({'error': 'Rover not connected'}), 400

    try:
        rover_url = f"http://{rover_state['ip_address']}:8080/frame"
        response = requests.get(rover_url, timeout=5)

        if response.status_code == 200:
            return response.content, 200, {'Content-Type': 'image/jpeg'}
        else:
            return jsonify({'error': 'Failed to get snapshot'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== MAP & NAVIGATION ====================

@app.route('/api/map/keyframes', methods=['GET'])
def get_keyframes():
    """Get all keyframes for map visualization"""
    try:
        keyframes = map_api.get_keyframes()
        return jsonify({'success': True, 'keyframes': keyframes})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/map/features', methods=['GET'])
def get_map_features():
    """Get map feature points for visualization"""
    try:
        features = map_api.get_features()
        return jsonify({'success': True, 'features': features})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/navigation/goto', methods=['POST'])
def navigate_to_keyframe():
    """Navigate to specific keyframe"""
    if not rover_state['connected'] or not rover_api:
        return jsonify({'success': False, 'message': 'Rover not connected'}), 400

    data = request.json
    target_keyframe = data.get('keyframe_id')

    try:
        # Start navigation in background thread
        rover_state['status'] = 'navigating'
        threading.Thread(
            target=rover_api.navigate_to_keyframe,
            args=(target_keyframe,),
            daemon=True
        ).start()

        socketio.emit('navigation_started', {'target': target_keyframe})

        return jsonify({'success': True, 'message': f'Navigating to keyframe {target_keyframe}'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# ==================== PATIENT MANAGEMENT ====================

@app.route('/api/patients', methods=['GET'])
def get_patients():
    """Get all patients"""
    patients = patient_api.get_all_patients()
    return jsonify({'success': True, 'patients': patients})

@app.route('/api/patients/<patient_id>', methods=['GET'])
def get_patient(patient_id):
    """Get specific patient"""
    patient = patient_api.get_patient(patient_id)
    if patient:
        return jsonify({'success': True, 'patient': patient})
    else:
        return jsonify({'success': False, 'message': 'Patient not found'}), 404

@app.route('/api/patients', methods=['POST'])
def add_patient():
    """Add new patient"""
    data = request.json
    try:
        patient = patient_api.add_patient(data)
        return jsonify({'success': True, 'patient': patient})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 400

@app.route('/api/patients/<patient_id>', methods=['PUT'])
def update_patient(patient_id):
    """Update patient information"""
    data = request.json
    try:
        patient = patient_api.update_patient(patient_id, data)
        if patient:
            return jsonify({'success': True, 'patient': patient})
        else:
            return jsonify({'success': False, 'message': 'Patient not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 400

@app.route('/api/patients/<patient_id>', methods=['DELETE'])
def delete_patient(patient_id):
    """Delete patient"""
    success = patient_api.delete_patient(patient_id)
    if success:
        return jsonify({'success': True, 'message': 'Patient deleted'})
    else:
        return jsonify({'success': False, 'message': 'Patient not found'}), 404

@app.route('/api/patients/<patient_id>/deliver', methods=['POST'])
def deliver_to_patient(patient_id):
    """Start delivery to patient"""
    if not rover_state['connected']:
        return jsonify({'success': False, 'message': 'Rover not connected'}), 400

    patient = patient_api.get_patient(patient_id)
    if not patient:
        return jsonify({'success': False, 'message': 'Patient not found'}), 404

    keyframe_id = patient.get('keyframe_id')
    if keyframe_id is None:
        return jsonify({'success': False, 'message': 'Patient has no assigned location'}), 400

    try:
        # Start delivery
        rover_state['status'] = 'delivering'
        threading.Thread(
            target=rover_api.navigate_to_keyframe,
            args=(keyframe_id,),
            kwargs={'patient_id': patient_id},
            daemon=True
        ).start()

        socketio.emit('delivery_started', {
            'patient_id': patient_id,
            'patient_name': patient['name'],
            'keyframe_id': keyframe_id
        })

        return jsonify({
            'success': True,
            'message': f'Delivering medication to {patient["name"]}',
            'keyframe_id': keyframe_id
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# ==================== WEBSOCKET EVENTS ====================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('rover_state', rover_state)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')

@socketio.on('request_status')
def handle_status_request():
    """Handle status request from client"""
    emit('rover_state', rover_state)

# ==================== BACKGROUND TASKS ====================

def update_rover_status():
    """Background thread to update rover status periodically"""
    while rover_state['connected']:
        try:
            if rover_api:
                # Get current position from navigation system
                # This would integrate with navig.py

                rover_state['last_update'] = datetime.now().isoformat()

                # Emit status update via WebSocket
                socketio.emit('rover_state', rover_state)

        except Exception as e:
            print(f"Error updating rover status: {e}")

        time.sleep(1)  # Update every second

# ==================== MAIN ====================

if __name__ == '__main__':
    print("=" * 60)
    print("MEDIBOT - Medication Delivery Rover System")
    print("=" * 60)
    print(f"Backend API starting on http://0.0.0.0:5000")
    print(f"Frontend available at http://localhost:5000")
    print("=" * 60)

    # Run with SocketIO
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
