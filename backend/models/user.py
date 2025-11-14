"""
User Model
Data model and manager for user authentication
"""

import json
import os
import uuid
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional


class User:
    """User data model"""

    def __init__(self, user_id: str, email: str, password_hash: str, 
                 full_name: str, created_at: Optional[str] = None,
                 is_verified: bool = False, last_login: Optional[str] = None):
        self.user_id = user_id
        self.email = email.lower()
        self.password_hash = password_hash
        self.full_name = full_name
        self.created_at = created_at or datetime.now().isoformat()
        self.is_verified = is_verified
        self.last_login = last_login
        self.failed_login_attempts = 0
        self.locked_until = None

    def to_dict(self) -> Dict:
        """Convert user to dictionary"""
        return {
            'user_id': self.user_id,
            'email': self.email,
            'password_hash': self.password_hash,
            'full_name': self.full_name,
            'created_at': self.created_at,
            'is_verified': self.is_verified,
            'last_login': self.last_login,
            'failed_login_attempts': self.failed_login_attempts,
            'locked_until': self.locked_until
        }

    @staticmethod
    def from_dict(data: Dict) -> 'User':
        """Create user from dictionary"""
        user = User(
            user_id=data['user_id'],
            email=data['email'],
            password_hash=data['password_hash'],
            full_name=data['full_name'],
            created_at=data.get('created_at'),
            is_verified=data.get('is_verified', False),
            last_login=data.get('last_login')
        )
        user.failed_login_attempts = data.get('failed_login_attempts', 0)
        user.locked_until = data.get('locked_until')
        return user

    def check_password(self, password: str) -> bool:
        """Verify password against hash"""
        # Simple hash check - in production, use bcrypt
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        return password_hash == self.password_hash

    def is_locked(self) -> bool:
        """Check if account is locked"""
        if self.locked_until:
            lock_time = datetime.fromisoformat(self.locked_until)
            if datetime.now() < lock_time:
                return True
            else:
                self.locked_until = None
                self.failed_login_attempts = 0
        return False


class OTPManager:
    """Manages OTP generation and verification"""

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            base_dir = os.path.dirname(base_dir)
            data_dir = os.path.join(base_dir, 'data', 'auth')
        
        self.data_dir = data_dir
        self.otp_file = os.path.join(data_dir, 'otps.json')
        self.otps = {}
        self.ensure_data_file()
        self.load_otps()

    def ensure_data_file(self):
        """Ensure data directory and file exist"""
        os.makedirs(self.data_dir, exist_ok=True)
        if not os.path.exists(self.otp_file):
            with open(self.otp_file, 'w') as f:
                json.dump({}, f)

    def load_otps(self):
        """Load OTPs from file"""
        try:
            with open(self.otp_file, 'r') as f:
                self.otps = json.load(f)
            # Clean expired OTPs
            self.clean_expired()
        except Exception as e:
            print(f"Error loading OTPs: {e}")
            self.otps = {}

    def save_otps(self):
        """Save OTPs to file"""
        try:
            with open(self.otp_file, 'w') as f:
                json.dump(self.otps, f, indent=2)
        except Exception as e:
            print(f"Error saving OTPs: {e}")

    def generate_otp(self, email: str) -> str:
        """Generate a 6-digit OTP for email"""
        otp = ''.join([str(secrets.randbelow(10)) for _ in range(6)])
        expires_at = (datetime.now() + timedelta(minutes=10)).isoformat()
        
        self.otps[email] = {
            'otp': otp,
            'expires_at': expires_at,
            'attempts': 0
        }
        self.save_otps()
        return otp

    def verify_otp(self, email: str, otp: str) -> bool:
        """Verify OTP for email"""
        if email not in self.otps:
            return False

        otp_data = self.otps[email]
        
        # Check expiration
        expires_at = datetime.fromisoformat(otp_data['expires_at'])
        if datetime.now() > expires_at:
            del self.otps[email]
            self.save_otps()
            return False

        # Check attempts
        if otp_data['attempts'] >= 3:
            return False

        # Check OTP
        if otp_data['otp'] == otp:
            del self.otps[email]
            self.save_otps()
            return True
        else:
            otp_data['attempts'] += 1
            self.save_otps()
            return False

    def clean_expired(self):
        """Remove expired OTPs"""
        now = datetime.now()
        expired = [
            email for email, data in self.otps.items()
            if datetime.fromisoformat(data['expires_at']) < now
        ]
        for email in expired:
            del self.otps[email]
        if expired:
            self.save_otps()


class UserManager:
    """Manages user data persistence and operations"""

    def __init__(self, data_file: str = None):
        if data_file is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            base_dir = os.path.dirname(base_dir)
            data_file = os.path.join(base_dir, 'data', 'auth', 'users.json')
        
        self.data_file = data_file
        self.users: Dict[str, User] = {}
        self.ensure_data_file()
        self.load_users()

    def ensure_data_file(self):
        """Ensure data directory and file exist"""
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        if not os.path.exists(self.data_file):
            with open(self.data_file, 'w') as f:
                json.dump([], f)

    def load_users(self):
        """Load users from JSON file"""
        try:
            with open(self.data_file, 'r') as f:
                users_data = json.load(f)

            self.users = {
                u['email']: User.from_dict(u)
                for u in users_data
            }

            print(f"Loaded {len(self.users)} users")

        except Exception as e:
            print(f"Error loading users: {e}")
            self.users = {}

    def save_users(self):
        """Save users to JSON file"""
        try:
            users_data = [u.to_dict() for u in self.users.values()]
            with open(self.data_file, 'w') as f:
                json.dump(users_data, f, indent=2)
        except Exception as e:
            print(f"Error saving users: {e}")

    def create_user(self, email: str, password: str, full_name: str) -> User:
        """Create a new user"""
        email = email.lower()
        
        if email in self.users:
            raise ValueError("User already exists")

        # Hash password
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        user_id = str(uuid.uuid4())
        user = User(user_id, email, password_hash, full_name)
        
        self.users[email] = user
        self.save_users()
        
        return user

    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        return self.users.get(email.lower())

    def verify_user(self, email: str):
        """Mark user as verified"""
        user = self.get_user_by_email(email)
        if user:
            user.is_verified = True
            self.save_users()

    def update_last_login(self, email: str):
        """Update last login timestamp"""
        user = self.get_user_by_email(email)
        if user:
            user.last_login = datetime.now().isoformat()
            user.failed_login_attempts = 0
            user.locked_until = None
            self.save_users()

    def record_failed_login(self, email: str):
        """Record failed login attempt"""
        user = self.get_user_by_email(email)
        if user:
            user.failed_login_attempts += 1
            if user.failed_login_attempts >= 5:
                user.locked_until = (datetime.now() + timedelta(minutes=30)).isoformat()
            self.save_users()
