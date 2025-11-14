"""
Authentication API Module
Handles user authentication, registration, and OTP verification
"""

from flask import session
from typing import Dict, Optional
from datetime import datetime, timedelta
import secrets


class AuthAPI:
    """Interface for authentication operations"""

    def __init__(self, user_manager, otp_manager, email_service):
        self.user_manager = user_manager
        self.otp_manager = otp_manager
        self.email_service = email_service
        self.active_sessions = {}  # In-memory session store

    def register(self, email: str, password: str, full_name: str) -> Dict:
        """
        Register a new user
        
        Args:
            email: User's email address
            password: User's password
            full_name: User's full name
            
        Returns:
            Dict with status and message
        """
        try:
            # Validate input
            if not email or not password or not full_name:
                return {'success': False, 'message': 'All fields are required'}

            if len(password) < 8:
                return {'success': False, 'message': 'Password must be at least 8 characters'}

            # Create user
            user = self.user_manager.create_user(email, password, full_name)

            # Generate OTP
            otp = self.otp_manager.generate_otp(email)

            # Send OTP email
            self.email_service.send_otp_email(email, otp, full_name)

            return {
                'success': True,
                'message': 'Registration successful! Please check your email for the verification code.',
                'requires_otp': True,
                'email': email
            }

        except ValueError as e:
            return {'success': False, 'message': str(e)}
        except Exception as e:
            print(f"Registration error: {e}")
            return {'success': False, 'message': 'Registration failed. Please try again.'}

    def verify_registration(self, email: str, otp: str) -> Dict:
        """
        Verify user registration with OTP
        
        Args:
            email: User's email address
            otp: OTP code sent to email
            
        Returns:
            Dict with status and message
        """
        try:
            # Verify OTP
            if not self.otp_manager.verify_otp(email, otp):
                return {'success': False, 'message': 'Invalid or expired verification code'}

            # Mark user as verified
            self.user_manager.verify_user(email)

            # Send welcome email
            user = self.user_manager.get_user_by_email(email)
            if user:
                self.email_service.send_welcome_email(email, user.full_name)

            return {
                'success': True,
                'message': 'Account verified successfully! You can now log in.'
            }

        except Exception as e:
            print(f"Verification error: {e}")
            return {'success': False, 'message': 'Verification failed. Please try again.'}

    def login_step1(self, email: str, password: str) -> Dict:
        """
        First step of login - verify email and password
        
        Args:
            email: User's email address
            password: User's password
            
        Returns:
            Dict with status and message
        """
        try:
            # Get user
            user = self.user_manager.get_user_by_email(email)
            
            if not user:
                return {'success': False, 'message': 'Invalid email or password'}

            # Check if account is locked
            if user.is_locked():
                return {
                    'success': False,
                    'message': 'Account is temporarily locked due to multiple failed login attempts. Please try again later.'
                }

            # Check if user is verified
            if not user.is_verified:
                return {
                    'success': False,
                    'message': 'Please verify your email address before logging in.'
                }

            # Verify password
            if not user.check_password(password):
                self.user_manager.record_failed_login(email)
                return {'success': False, 'message': 'Invalid email or password'}

            # Generate OTP for second factor
            otp = self.otp_manager.generate_otp(email)

            # Send OTP email
            self.email_service.send_otp_email(email, otp, user.full_name)

            return {
                'success': True,
                'message': 'Password verified! Please check your email for the verification code.',
                'requires_otp': True,
                'email': email
            }

        except Exception as e:
            print(f"Login error: {e}")
            return {'success': False, 'message': 'Login failed. Please try again.'}

    def login_step2(self, email: str, otp: str) -> Dict:
        """
        Second step of login - verify OTP
        
        Args:
            email: User's email address
            otp: OTP code sent to email
            
        Returns:
            Dict with status, message, and session token
        """
        try:
            # Verify OTP
            if not self.otp_manager.verify_otp(email, otp):
                return {'success': False, 'message': 'Invalid or expired verification code'}

            # Update last login
            self.user_manager.update_last_login(email)

            # Create session
            session_token = secrets.token_urlsafe(32)
            user = self.user_manager.get_user_by_email(email)
            
            self.active_sessions[session_token] = {
                'email': email,
                'user_id': user.user_id,
                'full_name': user.full_name,
                'created_at': datetime.now().isoformat(),
                'expires_at': (datetime.now() + timedelta(hours=24)).isoformat()
            }

            return {
                'success': True,
                'message': 'Login successful!',
                'session_token': session_token,
                'user': {
                    'email': user.email,
                    'full_name': user.full_name,
                    'user_id': user.user_id
                }
            }

        except Exception as e:
            print(f"Login verification error: {e}")
            return {'success': False, 'message': 'Login failed. Please try again.'}

    def logout(self, session_token: str) -> Dict:
        """
        Logout user
        
        Args:
            session_token: User's session token
            
        Returns:
            Dict with status and message
        """
        try:
            if session_token in self.active_sessions:
                del self.active_sessions[session_token]
            
            return {'success': True, 'message': 'Logged out successfully'}

        except Exception as e:
            print(f"Logout error: {e}")
            return {'success': False, 'message': 'Logout failed'}

    def verify_session(self, session_token: str) -> Optional[Dict]:
        """
        Verify if session is valid
        
        Args:
            session_token: Session token to verify
            
        Returns:
            Session data if valid, None otherwise
        """
        if session_token not in self.active_sessions:
            return None

        session_data = self.active_sessions[session_token]
        
        # Check expiration
        expires_at = datetime.fromisoformat(session_data['expires_at'])
        if datetime.now() > expires_at:
            del self.active_sessions[session_token]
            return None

        return session_data

    def resend_otp(self, email: str) -> Dict:
        """
        Resend OTP to user's email
        
        Args:
            email: User's email address
            
        Returns:
            Dict with status and message
        """
        try:
            user = self.user_manager.get_user_by_email(email)
            
            if not user:
                return {'success': False, 'message': 'User not found'}

            # Generate new OTP
            otp = self.otp_manager.generate_otp(email)

            # Send OTP email
            self.email_service.send_otp_email(email, otp, user.full_name)

            return {
                'success': True,
                'message': 'Verification code sent! Please check your email.'
            }

        except Exception as e:
            print(f"Resend OTP error: {e}")
            return {'success': False, 'message': 'Failed to resend code. Please try again.'}
