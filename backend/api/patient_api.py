"""
Patient API Module
Handles patient management operations
"""

from typing import Dict, List, Optional


class PatientAPI:
    """Interface for patient management"""

    def __init__(self, patient_manager):
        self.patient_manager = patient_manager

    def get_all_patients(self) -> List[Dict]:
        """Get all patients"""
        return self.patient_manager.get_all_patients()

    def get_patient(self, patient_id: str) -> Optional[Dict]:
        """Get specific patient by ID"""
        return self.patient_manager.get_patient(patient_id)

    def add_patient(self, patient_data: Dict) -> Dict:
        """Add new patient"""
        required_fields = ['name', 'room_number', 'medication']

        # Validate required fields
        for field in required_fields:
            if field not in patient_data:
                raise ValueError(f"Missing required field: {field}")

        return self.patient_manager.add_patient(patient_data)

    def update_patient(self, patient_id: str, patient_data: Dict) -> Optional[Dict]:
        """Update patient information"""
        return self.patient_manager.update_patient(patient_id, patient_data)

    def delete_patient(self, patient_id: str) -> bool:
        """Delete patient"""
        return self.patient_manager.delete_patient(patient_id)

    def get_patients_by_room(self, room_number: str) -> List[Dict]:
        """Get all patients in a specific room"""
        all_patients = self.get_all_patients()
        return [p for p in all_patients if p.get('room_number') == room_number]

    def get_patients_by_keyframe(self, keyframe_id: int) -> List[Dict]:
        """Get all patients at a specific keyframe location"""
        all_patients = self.get_all_patients()
        return [p for p in all_patients if p.get('keyframe_id') == keyframe_id]
