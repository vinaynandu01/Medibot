"""
Face Recognition API Routes
Handles patient face registration and recognition
"""

from flask import Blueprint, request, jsonify
import cv2
import numpy as np
import base64

face_bp = Blueprint('face', __name__)

def setup_face_routes(app, face_service, patient_manager):
    """Setup face recognition routes"""
    
    @app.route('/api/face/register', methods=['POST'])
    def register_patient_face():
        """
        Register a patient's face with multiple images
        Expects: patient_id, images (5 base64 encoded images)
        """
        if not face_service.is_available():
            return jsonify({
                'success': False,
                'message': 'Face recognition service not available'
            }), 503
        
        try:
            data = request.json
            patient_id = data.get('patient_id')
            
            if not patient_id:
                return jsonify({
                    'success': False,
                    'message': 'Patient ID required'
                }), 400
            
            # Get patient details
            patient = patient_manager.get_patient(patient_id)
            if not patient:
                return jsonify({
                    'success': False,
                    'message': 'Patient not found'
                }), 404
            
            # Decode images
            images = []
            for i in range(5):
                image_data = data.get(f'image{i}')
                if not image_data:
                    continue
                
                # Remove data URL prefix if present
                if 'base64,' in image_data:
                    image_data = image_data.split('base64,')[1]
                
                # Decode base64
                image_bytes = base64.b64decode(image_data)
                image_array = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                
                if image is not None:
                    images.append(image)
            
            if len(images) < 3:
                return jsonify({
                    'success': False,
                    'message': f'Need at least 3 valid images, received {len(images)}'
                }), 400
            
            # Register face
            result = face_service.register_patient_face(
                patient_id,
                patient['name'],
                images
            )
            
            if result['success']:
                # Update patient record with face data
                update_result = patient_manager.update_patient(patient_id, {
                    'face_registered': True,
                    'face_image': result.get('face_image'),
                    'face_embedding': face_service.face_data[
                        face_service.patient_ids.index(patient_id)
                    ].tolist()
                })
                print(f"✅ Updated patient {patient_id} with face data")
                print(f"   Face registered: {update_result.get('face_registered')}")
            
            return jsonify(result), 200 if result['success'] else 400
            
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Error processing face registration: {str(e)}'
            }), 500
    
    @app.route('/api/face/recognize', methods=['POST'])
    def recognize_face():
        """
        Recognize face in a single image
        Expects: image (base64 encoded)
        Returns: List of recognized patients
        """
        if not face_service.is_available():
            return jsonify({
                'success': False,
                'message': 'Face recognition service not available'
            }), 503
        
        try:
            data = request.json
            image_data = data.get('image')
            
            if not image_data:
                return jsonify({
                    'success': False,
                    'message': 'Image required'
                }), 400
            
            # Remove data URL prefix if present
            if 'base64,' in image_data:
                image_data = image_data.split('base64,')[1]
            
            # Decode image
            image_bytes = base64.b64decode(image_data)
            image_array = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image is None:
                return jsonify({
                    'success': False,
                    'message': 'Invalid image'
                }), 400
            
            # Recognize faces
            results = face_service.recognize_face(image)
            
            return jsonify({
                'success': True,
                'recognized_patients': results
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Error recognizing face: {str(e)}'
            }), 500
    
    @app.route('/api/face/status', methods=['GET'])
    def face_recognition_status():
        """Get face recognition service status"""
        stats = face_service.get_statistics()
        return jsonify({
            'success': True,
            'status': stats
        })
    
    @app.route('/api/face/check-registration/<patient_id>', methods=['GET'])
    def check_face_registration(patient_id):
        """Check if patient has face registered"""
        patient = patient_manager.get_patient(patient_id)
        
        if not patient:
            return jsonify({
                'success': False,
                'message': 'Patient not found'
            }), 404
        
        return jsonify({
            'success': True,
            'patient_id': patient_id,
            'face_registered': patient.get('face_registered', False),
            'face_image': patient.get('face_image')
        })
    
    @app.route('/api/face/delete/<patient_id>', methods=['DELETE'])
    def delete_patient_face(patient_id):
        """Delete patient's face registration"""
        try:
            patient = patient_manager.get_patient(patient_id)
            if not patient:
                return jsonify({
                    'success': False,
                    'message': 'Patient not found'
                }), 404
            
            # Remove from face service
            if patient_id in face_service.patient_ids:
                idx = face_service.patient_ids.index(patient_id)
                face_service.patient_ids.pop(idx)
                face_service.patient_names.pop(idx)
                face_service.face_data = np.delete(face_service.face_data, idx, axis=0)
                print(f"✅ Removed {patient_id} from face recognition service")
            
            # Update patient record
            update_result = patient_manager.update_patient(patient_id, {
                'face_registered': False,
                'face_image': None,
                'face_embedding': None
            })
            
            return jsonify({
                'success': True,
                'message': 'Face registration deleted successfully'
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Error deleting face registration: {str(e)}'
            }), 500
