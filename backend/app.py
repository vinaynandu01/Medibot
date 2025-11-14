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
from api.auth_api import AuthAPI
from api.face_api import setup_face_routes
from models.patient import PatientManager
from models.user import UserManager, OTPManager
from utils.config import Config
from utils.email_service import EmailService
from utils.face_recognition_service import FaceRecognitionService
from utils.navigation_controller import NavigationController

# Try to import email configuration
try:
    from utils.email_config import SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, FROM_NAME
    email_configured = True
except ImportError:
    print("⚠️  Warning: email_config.py not found. Using development mode.")
    print("   To enable email, copy email_config_template.py to email_config.py and configure it.")
    SMTP_HOST = 'smtp.gmail.com'
    SMTP_PORT = 587
    SMTP_USER = None
    SMTP_PASSWORD = None
    FROM_NAME = 'Medibot'
    email_configured = False

app = Flask(__name__, static_folder='../frontend', static_url_path='')
app.secret_key = os.urandom(24)  # Secret key for sessions
CORS(app, supports_credentials=True)
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuration
config = Config()
patient_manager = PatientManager()
user_manager = UserManager()
otp_manager = OTPManager()
email_service = EmailService(
    smtp_host=SMTP_HOST,
    smtp_port=SMTP_PORT,
    smtp_user=SMTP_USER,
    smtp_password=SMTP_PASSWORD
)

if email_configured and SMTP_PASSWORD and SMTP_PASSWORD != 'your-16-char-app-password':
    print("✅ Email service configured and ready")
else:
    print("⚠️  Email service in DEVELOPMENT MODE - OTPs will print to console")
    print("   Configure email_config.py to send actual emails")

# Initialize face recognition service
face_service = FaceRecognitionService()

# Initialize API modules
rover_api = None  # Will be initialized when rover is connected
patient_api = PatientAPI(patient_manager)
map_api = MapAPI()
auth_api = AuthAPI(user_manager, otp_manager, email_service)

# Setup face recognition routes
setup_face_routes(app, face_service, patient_manager)

# Initialize navigation controller
nav_controller = NavigationController(face_service, socketio)

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

# ==================== AUTHENTICATION MIDDLEWARE ====================

def require_auth(f):
    """Decorator to require authentication"""
    from functools import wraps
    
    @wraps(f)
    def decorated_function(*args, **kwargs):
        session_token = request.headers.get('Authorization')
        if session_token:
            session_token = session_token.replace('Bearer ', '')
        
        session_data = auth_api.verify_session(session_token)
        if not session_data:
            return jsonify({'success': False, 'message': 'Unauthorized'}), 401
        
        request.user = session_data
        return f(*args, **kwargs)
    
    return decorated_function

# ==================== STATIC FILE SERVING ====================

@app.route('/')
def index():
    """Serve the main frontend page - redirect to login"""
    return send_from_directory(app.static_folder, 'login.html')

@app.route('/dashboard')
def dashboard():
    """Serve the dashboard page"""
    return send_from_directory(app.static_folder, 'dashboard.html')

@app.route('/patients_list.html')
def patients_list():
    """Serve the patients list page"""
    return send_from_directory(app.static_folder, 'patients_list.html')

@app.route('/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory(app.static_folder, path)

# ==================== AUTHENTICATION ROUTES ====================

@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register a new user"""
    data = request.json
    result = auth_api.register(
        data.get('email'),
        data.get('password'),
        data.get('full_name')
    )
    return jsonify(result), 200 if result['success'] else 400

@app.route('/api/auth/verify-registration', methods=['POST'])
def verify_registration():
    """Verify user registration with OTP"""
    data = request.json
    result = auth_api.verify_registration(
        data.get('email'),
        data.get('otp')
    )
    return jsonify(result), 200 if result['success'] else 400

@app.route('/api/auth/login', methods=['POST'])
def login():
    """Login step 1 - verify email and password"""
    data = request.json
    result = auth_api.login_step1(
        data.get('email'),
        data.get('password')
    )
    return jsonify(result), 200 if result['success'] else 400

@app.route('/api/auth/verify-login', methods=['POST'])
def verify_login():
    """Login step 2 - verify OTP"""
    data = request.json
    result = auth_api.login_step2(
        data.get('email'),
        data.get('otp')
    )
    return jsonify(result), 200 if result['success'] else 400

@app.route('/api/auth/logout', methods=['POST'])
def logout():
    """Logout user"""
    session_token = request.headers.get('Authorization', '').replace('Bearer ', '')
    result = auth_api.logout(session_token)
    return jsonify(result)

@app.route('/api/auth/resend-otp', methods=['POST'])
def resend_otp():
    """Resend OTP"""
    data = request.json
    result = auth_api.resend_otp(data.get('email'))
    return jsonify(result), 200 if result['success'] else 400

@app.route('/api/auth/verify-session', methods=['GET'])
def verify_session():
    """Verify if session is valid"""
    session_token = request.headers.get('Authorization', '').replace('Bearer ', '')
    session_data = auth_api.verify_session(session_token)
    
    if session_data:
        return jsonify({'success': True, 'user': session_data})
    else:
        return jsonify({'success': False, 'message': 'Invalid session'}), 401

# ==================== ROVER CONNECTION ====================

@app.route('/api/rover/connect', methods=['POST'])
@require_auth
def connect_rover():
    """Connect to the rover with given IP address"""
    global rover_api, rover_state

    data = request.json
    ip_address = data.get('ip_address', '192.168.1.100')

    print(f"\n🔌 Attempting to connect to rover at {ip_address}:8080")

    try:
        # Test connection to rover with multiple endpoints
        test_urls = [
            f'http://{ip_address}:8080/queue/status',
            f'http://{ip_address}:8080/',
            f'http://{ip_address}:5000/queue/status',
            f'http://{ip_address}:5000/'
        ]
        
        connected = False
        working_url = None
        working_port = None
        
        for url in test_urls:
            try:
                print(f"   Testing: {url}")
                response = requests.get(url, timeout=3)
                if response.status_code == 200:
                    connected = True
                    working_url = url
                    # Extract port from URL
                    import re
                    port_match = re.search(r':(\d+)/', url)
                    working_port = int(port_match.group(1)) if port_match else 8080
                    print(f"   ✅ Connection successful!")
                    break
            except requests.exceptions.Timeout:
                print(f"   ⏱️  Timeout")
                continue
            except requests.exceptions.ConnectionError:
                print(f"   ❌ Connection refused")
                continue
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
                continue

        if connected:
            rover_state['connected'] = True
            rover_state['ip_address'] = ip_address
            rover_state['last_update'] = datetime.now().isoformat()

            # Initialize rover API with working port
            rover_api = RoverAPI(ip_address, port=working_port)

            # Start background thread for rover status updates
            threading.Thread(target=update_rover_status, daemon=True).start()

            # Emit connection status via WebSocket
            socketio.emit('rover_connected', {'ip_address': ip_address})

            return jsonify({
                'success': True,
                'message': f'Connected to rover at {ip_address}:{working_port}',
                'rover_state': rover_state,
                'port': working_port
            })
        else:
            error_msg = f'Cannot connect to rover at {ip_address}. Please check:\n'
            error_msg += '1. Rover is powered on\n'
            error_msg += '2. Rover is on the same network\n'
            error_msg += '3. IP address is correct\n'
            error_msg += '4. Rover server is running (port 5000 or 8080)'
            print(f"\n❌ {error_msg}")
            return jsonify({'success': False, 'message': error_msg}), 400

    except Exception as e:
        error_msg = f'Connection error: {str(e)}'
        print(f"\n❌ {error_msg}")
        return jsonify({'success': False, 'message': error_msg}), 500

@app.route('/api/rover/test', methods=['POST'])
@require_auth
def test_rover_connection():
    """Test connectivity to rover without connecting"""
    data = request.json
    ip_address = data.get('ip_address', '192.168.1.100')
    
    print(f"\n🔍 Testing connectivity to {ip_address}...")
    
    results = {
        'ip_address': ip_address,
        'reachable': False,
        'tests': []
    }
    
    # Test different ports and endpoints
    test_configs = [
        {'port': 8080, 'path': '/queue/status', 'name': 'Queue Status (8080)'},
        {'port': 8080, 'path': '/', 'name': 'Root Endpoint (8080)'},
        {'port': 5000, 'path': '/queue/status', 'name': 'Queue Status (5000)'},
        {'port': 5000, 'path': '/', 'name': 'Root Endpoint (5000)'},
    ]
    
    for config in test_configs:
        url = f"http://{ip_address}:{config['port']}{config['path']}"
        test_result = {
            'name': config['name'],
            'url': url,
            'status': 'testing'
        }
        
        try:
            response = requests.get(url, timeout=2)
            test_result['status'] = 'success'
            test_result['status_code'] = response.status_code
            test_result['response_time'] = response.elapsed.total_seconds()
            results['reachable'] = True
            print(f"   ✅ {config['name']}: {response.status_code}")
        except requests.exceptions.Timeout:
            test_result['status'] = 'timeout'
            test_result['error'] = 'Connection timeout (2s)'
            print(f"   ⏱️  {config['name']}: Timeout")
        except requests.exceptions.ConnectionError:
            test_result['status'] = 'refused'
            test_result['error'] = 'Connection refused'
            print(f"   ❌ {config['name']}: Connection refused")
        except Exception as e:
            test_result['status'] = 'error'
            test_result['error'] = str(e)
            print(f"   ❌ {config['name']}: {str(e)}")
        
        results['tests'].append(test_result)
    
    return jsonify(results)

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
@require_auth
def deliver_to_patient(patient_id):
    """Start delivery to patient with navigation and face recognition"""
    if not rover_state['connected']:
        return jsonify({'success': False, 'message': 'Rover not connected'}), 400

    patient = patient_api.get_patient(patient_id)
    if not patient:
        return jsonify({'success': False, 'message': 'Patient not found'}), 404

    keyframe_id = patient.get('keyframe_id')
    if keyframe_id is None:
        return jsonify({'success': False, 'message': 'Patient has no assigned location'}), 400

    try:
        # Start navigation with face recognition
        rover_state['status'] = 'delivering'
        
        result = nav_controller.start_navigation(
            patient_id=patient_id,
            patient_name=patient['name'],
            target_keyframe=keyframe_id
        )
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# ==================== NAVIGATION WITH FACE RECOGNITION ====================

@app.route('/api/navigation/start', methods=['POST'])
@require_auth
def start_navigation():
    """Start navigation to patient location with face recognition"""
    data = request.json
    patient_id = data.get('patient_id')
    
    if not patient_id:
        return jsonify({'success': False, 'message': 'Patient ID required'}), 400
    
    patient = patient_api.get_patient(patient_id)
    if not patient:
        return jsonify({'success': False, 'message': 'Patient not found'}), 404
    
    keyframe_id = patient.get('keyframe_id')
    if keyframe_id is None:
        return jsonify({'success': False, 'message': 'Patient has no assigned location'}), 400
    
    rover_state['status'] = 'navigating'
    result = nav_controller.start_navigation(
        patient_id=patient_id,
        patient_name=patient['name'],
        target_keyframe=keyframe_id
    )
    
    return jsonify(result)

@app.route('/api/navigation/stop', methods=['POST'])
@require_auth
def stop_navigation():
    """Emergency stop navigation"""
    result = nav_controller.stop_navigation()
    rover_state['status'] = 'idle'
    return jsonify(result)

@app.route('/api/navigation/status', methods=['GET'])
def get_navigation_status():
    """Get current navigation status"""
    status = nav_controller.get_status()
    return jsonify({'success': True, 'status': status})

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
    
    # Load patient face embeddings
    if face_service.is_available():
        face_service.load_patient_faces_from_file(patient_manager)
    
    # Run with SocketIO
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
