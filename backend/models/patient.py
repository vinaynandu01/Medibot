"""
Patient Model
Data model and manager for patient information
"""

import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional


class Patient:
    """Patient data model"""

    def __init__(self, patient_id: str, name: str, room_number: str,
                 medication: str, keyframe_id: Optional[int] = None,
                 schedule: Optional[str] = None, notes: Optional[str] = None):
        self.patient_id = patient_id
        self.name = name
        self.room_number = room_number
        self.medication = medication
        self.keyframe_id = keyframe_id
        self.schedule = schedule or "3 times daily"
        self.notes = notes or ""
        self.created_at = datetime.now().isoformat()
        self.last_delivery = None
        self.delivery_count = 0
        self.face_registered = False
        self.face_image = None
        self.face_embedding = None

    def to_dict(self) -> Dict:
        """Convert patient to dictionary"""
        return {
            'patient_id': self.patient_id,
            'name': self.name,
            'room_number': self.room_number,
            'medication': self.medication,
            'keyframe_id': self.keyframe_id,
            'schedule': self.schedule,
            'notes': self.notes,
            'created_at': self.created_at,
            'last_delivery': self.last_delivery,
            'delivery_count': self.delivery_count,
            'face_registered': self.face_registered,
            'face_image': self.face_image,
            'face_embedding': self.face_embedding
        }

    @classmethod
    def from_dict(cls, data: Dict):
        """Create patient from dictionary"""
        patient = cls(
            patient_id=data['patient_id'],
            name=data['name'],
            room_number=data['room_number'],
            medication=data['medication'],
            keyframe_id=data.get('keyframe_id'),
            schedule=data.get('schedule'),
            notes=data.get('notes')
        )
        patient.created_at = data.get('created_at', datetime.now().isoformat())
        patient.last_delivery = data.get('last_delivery')
        patient.delivery_count = data.get('delivery_count', 0)
        patient.face_registered = data.get('face_registered', False)
        patient.face_image = data.get('face_image')
        patient.face_embedding = data.get('face_embedding')
        return patient


class PatientManager:
    """Manages patient data persistence and operations"""

    def __init__(self, data_file: str = None):
        if data_file is None:
            # Get the base directory (Medibot root)
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            base_dir = os.path.dirname(base_dir)  # Go up one more level from backend
            data_file = os.path.join(base_dir, 'data', 'patients', 'patients.json')
        
        self.data_file = data_file
        self.patients: Dict[str, Patient] = {}
        self.ensure_data_file()
        self.load_patients()

    def ensure_data_file(self):
        """Ensure data directory and file exist"""
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        if not os.path.exists(self.data_file):
            with open(self.data_file, 'w') as f:
                json.dump([], f)

    def load_patients(self):
        """Load patients from JSON file"""
        try:
            with open(self.data_file, 'r') as f:
                patients_data = json.load(f)

            self.patients = {
                p['patient_id']: Patient.from_dict(p)
                for p in patients_data
            }

            print(f"Loaded {len(self.patients)} patients")

        except Exception as e:
            print(f"Error loading patients: {e}")
            self.patients = {}

    def save_patients(self):
        """Save patients to JSON file"""
        try:
            print(f"\n💾 SAVE_PATIENTS called")
            print(f"💾 Total patients: {len(self.patients)}")
            patients_data = [p.to_dict() for p in self.patients.values()]
            
            # Check if any patient has face data
            with_face = sum(1 for p in patients_data if p.get('face_registered'))
            print(f"💾 Patients with face_registered: {with_face}")
            
            print(f"💾 Writing to file: {self.data_file}")
            with open(self.data_file, 'w') as f:
                json.dump(patients_data, f, indent=2)
            print(f"✅ Successfully saved {len(patients_data)} patients")

        except Exception as e:
            print(f"❌ Error saving patients: {e}")
            import traceback
            traceback.print_exc()

    def add_patient(self, patient_data: Dict) -> Dict:
        """Add new patient"""
        patient_id = str(uuid.uuid4())[:8]

        patient = Patient(
            patient_id=patient_id,
            name=patient_data['name'],
            room_number=patient_data['room_number'],
            medication=patient_data['medication'],
            keyframe_id=patient_data.get('keyframe_id'),
            schedule=patient_data.get('schedule'),
            notes=patient_data.get('notes')
        )

        self.patients[patient_id] = patient
        self.save_patients()

        return patient.to_dict()

    def get_patient(self, patient_id: str) -> Optional[Dict]:
        """Get patient by ID"""
        patient = self.patients.get(patient_id)
        return patient.to_dict() if patient else None

    def get_all_patients(self) -> List[Dict]:
        """Get all patients"""
        return [p.to_dict() for p in self.patients.values()]

    def update_patient(self, patient_id: str, patient_data: Dict) -> Optional[Dict]:
        """Update patient information"""
        print(f"\n🔍 UPDATE_PATIENT called for patient_id: {patient_id}")
        print(f"🔍 patient_data keys: {list(patient_data.keys())}")
        
        if patient_id not in self.patients:
            print(f"❌ Patient {patient_id} not found in self.patients")
            return None

        patient = self.patients[patient_id]
        print(f"✅ Found patient: {patient.name}")

        # Update fields
        if 'name' in patient_data:
            patient.name = patient_data['name']
        if 'room_number' in patient_data:
            patient.room_number = patient_data['room_number']
        if 'medication' in patient_data:
            patient.medication = patient_data['medication']
        if 'keyframe_id' in patient_data:
            patient.keyframe_id = patient_data['keyframe_id']
        if 'schedule' in patient_data:
            patient.schedule = patient_data['schedule']
        if 'notes' in patient_data:
            patient.notes = patient_data['notes']
        if 'face_registered' in patient_data:
            print(f"📸 Setting face_registered = {patient_data['face_registered']}")
            patient.face_registered = patient_data['face_registered']
        if 'face_image' in patient_data:
            image_len = len(patient_data['face_image']) if patient_data['face_image'] else 0
            print(f"📸 Setting face_image (length: {image_len})")
            patient.face_image = patient_data['face_image']
        if 'face_embedding' in patient_data:
            embed_len = len(patient_data['face_embedding']) if patient_data['face_embedding'] else 0
            print(f"📸 Setting face_embedding (length: {embed_len})")
            patient.face_embedding = patient_data['face_embedding']

        print(f"💾 Calling save_patients()...")
        self.save_patients()
        result = patient.to_dict()
        print(f"✅ Patient updated, returning dict with face_registered={result.get('face_registered')}")
        return result

    def delete_patient(self, patient_id: str) -> bool:
        """Delete patient"""
        if patient_id in self.patients:
            del self.patients[patient_id]
            self.save_patients()
            return True
        return False

    def record_delivery(self, patient_id: str):
        """Record medication delivery"""
        if patient_id in self.patients:
            patient = self.patients[patient_id]
            patient.last_delivery = datetime.now().isoformat()
            patient.delivery_count += 1
            self.save_patients()

    def get_patients_needing_delivery(self) -> List[Dict]:
        """Get patients that need medication delivery"""
        # This could be enhanced with schedule logic
        return self.get_all_patients()
