"""
Face Recognition Service for Patient Identification
Uses MTCNN for face detection and FaceNet for embeddings
"""

import cv2
import numpy as np
import base64
import os
import json
from typing import Optional, Dict, List, Tuple
import threading
import time

try:
    from mtcnn.mtcnn import MTCNN
    from keras_facenet import FaceNet
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    print("⚠️  Face recognition libraries not installed. Install with: pip install mtcnn keras-facenet")
    FACE_RECOGNITION_AVAILABLE = False

class FaceRecognitionService:
    """Service for face detection, recognition, and patient verification"""
    
    def __init__(self):
        self.initialized = False
        self.detector = None
        self.embedder = None
        self.face_data = []  # List of embeddings
        self.patient_ids = []  # List of patient IDs
        self.patient_names = {}  # Dict mapping patient_id to name
        self.recognition_threshold = 0.7  # Cosine similarity threshold
        
        # Initialize models in background
        if FACE_RECOGNITION_AVAILABLE:
            threading.Thread(target=self._initialize_models, daemon=True).start()
    
    def _initialize_models(self):
        """Initialize MTCNN and FaceNet models"""
        try:
            print("🔄 Loading face recognition models...")
            self.detector = MTCNN()
            self.embedder = FaceNet()
            self.initialized = True
            print("✅ Face recognition models loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load face recognition models: {e}")
            self.initialized = False
    
    def is_available(self) -> bool:
        """Check if face recognition is available and initialized"""
        return FACE_RECOGNITION_AVAILABLE and self.initialized
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in image
        Returns list of face bounding boxes and confidence
        """
        if not self.is_available():
            return []
        
        try:
            faces = self.detector.detect_faces(image)
            return faces
        except Exception as e:
            print(f"Error detecting faces: {e}")
            return []
    
    def extract_face_embedding(self, image: np.ndarray, face_box: Dict) -> Optional[np.ndarray]:
        """
        Extract face embedding from detected face region
        """
        if not self.is_available():
            return None
        
        try:
            x, y, w, h = face_box['box']
            x, y = max(0, x), max(0, y)
            
            # Crop and resize face
            cropped_face = image[y:y+h, x:x+w]
            if cropped_face.size == 0:
                return None
            
            resized_face = cv2.resize(cropped_face, (160, 160))
            rgb_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2RGB)
            
            # Get embedding
            embedding = self.embedder.embeddings(np.expand_dims(rgb_face, axis=0)).flatten()
            return embedding
        except Exception as e:
            print(f"Error extracting face embedding: {e}")
            return None
    
    def register_patient_face(self, patient_id: str, patient_name: str, images: List[np.ndarray]) -> Dict:
        """
        Register a patient's face with multiple images
        Returns: {'success': bool, 'message': str, 'face_image': base64_str}
        """
        if not self.is_available():
            return {
                'success': False,
                'message': 'Face recognition not available'
            }
        
        if len(images) < 3:
            return {
                'success': False,
                'message': 'Please provide at least 3 images for registration'
            }
        
        embeddings = []
        stored_image = None
        
        for image in images:
            faces = self.detect_faces(image)
            
            if not faces:
                continue
            
            # Get embedding from first detected face
            embedding = self.extract_face_embedding(image, faces[0])
            if embedding is not None:
                embeddings.append(embedding)
                
                # Store first valid face image
                if stored_image is None:
                    x, y, w, h = faces[0]['box']
                    x, y = max(0, x), max(0, y)
                    face_img = image[y:y+h, x:x+w]
                    _, buffer = cv2.imencode('.jpg', face_img)
                    stored_image = base64.b64encode(buffer).decode('utf-8')
        
        if len(embeddings) < 3:
            return {
                'success': False,
                'message': f'Only detected {len(embeddings)} valid faces. Need at least 3.'
            }
        
        # Calculate mean embedding
        mean_embedding = np.mean(embeddings, axis=0)
        
        # Store in memory
        if patient_id in self.patient_ids:
            # Update existing
            idx = self.patient_ids.index(patient_id)
            self.face_data[idx] = mean_embedding
        else:
            # Add new
            self.face_data.append(mean_embedding)
            self.patient_ids.append(patient_id)
        
        self.patient_names[patient_id] = patient_name
        
        return {
            'success': True,
            'message': f'Successfully registered face for {patient_name}',
            'face_image': stored_image,
            'embeddings_count': len(embeddings)
        }
    
    def recognize_face(self, image: np.ndarray) -> List[Dict]:
        """
        Recognize faces in image and match with registered patients
        Returns: List of {'patient_id': str, 'patient_name': str, 'confidence': float, 'box': dict}
        """
        if not self.is_available():
            return []
        
        if len(self.face_data) == 0:
            return []
        
        faces = self.detect_faces(image)
        results = []
        
        for face in faces:
            embedding = self.extract_face_embedding(image, face)
            if embedding is None:
                continue
            
            # Compare with all stored embeddings
            similarities = []
            for stored_embedding in self.face_data:
                similarity = self._cosine_similarity(embedding, stored_embedding)
                similarities.append(similarity)
            
            # Find best match
            best_idx = np.argmax(similarities)
            best_similarity = similarities[best_idx]
            
            if best_similarity >= self.recognition_threshold:
                patient_id = self.patient_ids[best_idx]
                patient_name = self.patient_names.get(patient_id, 'Unknown')
                
                results.append({
                    'patient_id': patient_id,
                    'patient_name': patient_name,
                    'confidence': float(best_similarity),
                    'box': face['box']
                })
        
        return results
    
    def _cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def load_patient_faces_from_file(self, patient_manager):
        """Load patient face data from patient manager"""
        try:
            patients = patient_manager.get_all()
            loaded_count = 0
            
            for patient in patients:
                if 'face_embedding' in patient and patient['face_embedding']:
                    patient_id = patient['id']
                    patient_name = patient['name']
                    embedding = np.array(patient['face_embedding'])
                    
                    if patient_id not in self.patient_ids:
                        self.face_data.append(embedding)
                        self.patient_ids.append(patient_id)
                        self.patient_names[patient_id] = patient_name
                        loaded_count += 1
            
            if loaded_count > 0:
                print(f"✅ Loaded {loaded_count} patient face embeddings")
            
        except Exception as e:
            print(f"Error loading patient faces: {e}")
    
    def get_statistics(self) -> Dict:
        """Get face recognition statistics"""
        return {
            'available': self.is_available(),
            'initialized': self.initialized,
            'registered_patients': len(self.patient_ids),
            'threshold': self.recognition_threshold
        }
