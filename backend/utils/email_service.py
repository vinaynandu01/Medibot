"""
Email Service for OTP
Handles sending OTP emails using SMTP
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional


class EmailService:
    """Service for sending emails"""

    def __init__(self, smtp_host: str = 'smtp.gmail.com', smtp_port: int = 587,
                 smtp_user: Optional[str] = None, smtp_password: Optional[str] = None):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user or 'noreply@medibot.com'
        self.smtp_password = smtp_password
        self.from_email = smtp_user or 'Medibot <noreply@medibot.com>'

    def send_otp_email(self, to_email: str, otp: str, name: str = 'User') -> bool:
        """
        Send OTP verification email
        
        Args:
            to_email: Recipient email address
            otp: 6-digit OTP code
            name: User's name
            
        Returns:
            bool: True if email sent successfully
        """
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = 'Medibot - Your Verification Code'
            msg['From'] = self.from_email
            msg['To'] = to_email

            # Create HTML content
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        line-height: 1.6;
                        color: #333;
                        max-width: 600px;
                        margin: 0 auto;
                        padding: 20px;
                    }}
                    .header {{
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 30px;
                        text-align: center;
                        border-radius: 10px 10px 0 0;
                    }}
                    .content {{
                        background: #f9f9f9;
                        padding: 30px;
                        border: 1px solid #ddd;
                    }}
                    .otp-box {{
                        background: white;
                        border: 2px dashed #667eea;
                        padding: 20px;
                        text-align: center;
                        margin: 20px 0;
                        border-radius: 8px;
                    }}
                    .otp-code {{
                        font-size: 32px;
                        font-weight: bold;
                        color: #667eea;
                        letter-spacing: 8px;
                        font-family: 'Courier New', monospace;
                    }}
                    .footer {{
                        background: #333;
                        color: #999;
                        padding: 20px;
                        text-align: center;
                        font-size: 12px;
                        border-radius: 0 0 10px 10px;
                    }}
                    .warning {{
                        background: #fff3cd;
                        border-left: 4px solid #ffc107;
                        padding: 12px;
                        margin: 15px 0;
                        border-radius: 4px;
                    }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🤖 MEDIBOT</h1>
                    <p>Medication Delivery Rover System</p>
                </div>
                <div class="content">
                    <h2>Hello {name}!</h2>
                    <p>You've requested to verify your identity. Use the code below to complete your authentication:</p>
                    
                    <div class="otp-box">
                        <p style="margin: 0; color: #666;">Your Verification Code</p>
                        <div class="otp-code">{otp}</div>
                        <p style="margin: 10px 0 0 0; color: #999; font-size: 14px;">Valid for 10 minutes</p>
                    </div>
                    
                    <div class="warning">
                        <strong>⚠️ Security Notice:</strong> Never share this code with anyone. Medibot staff will never ask for your verification code.
                    </div>
                    
                    <p>If you didn't request this code, please ignore this email or contact support if you have concerns.</p>
                </div>
                <div class="footer">
                    <p>&copy; 2025 Medibot - Medication Delivery Rover System</p>
                    <p>This is an automated message, please do not reply.</p>
                </div>
            </body>
            </html>
            """

            # Create plain text version
            text_content = f"""
            MEDIBOT - Medication Delivery Rover System
            
            Hello {name}!
            
            Your verification code is: {otp}
            
            This code is valid for 10 minutes.
            
            If you didn't request this code, please ignore this email.
            
            ---
            © 2025 Medibot
            """

            # Attach both versions
            part1 = MIMEText(text_content, 'plain')
            part2 = MIMEText(html_content, 'html')
            msg.attach(part1)
            msg.attach(part2)

            # Send email
            if self.smtp_password and self.smtp_password != 'your-16-char-app-password':
                # Use actual SMTP server
                try:
                    with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                        server.starttls()
                        server.login(self.smtp_user, self.smtp_password)
                        server.send_message(msg)
                    print(f"✅ OTP email sent successfully to {to_email}")
                    return True
                except smtplib.SMTPAuthenticationError:
                    print(f"❌ Email authentication failed. Please check your credentials.")
                    print(f"   Gmail users: Make sure you're using an App Password, not your regular password")
                    print(f"   Visit: https://myaccount.google.com/apppasswords")
                    raise Exception("Email authentication failed. Please configure email settings.")
                except Exception as email_error:
                    print(f"❌ Failed to send email: {email_error}")
                    raise Exception(f"Failed to send email: {email_error}")
            else:
                # Development mode - print to console
                print(f"\n{'='*60}")
                print(f"⚠️  DEVELOPMENT MODE - Configure email in email_config.py")
                print(f"{'='*60}")
                print(f"OTP Email to: {to_email}")
                print(f"OTP Code: {otp}")
                print(f"{'='*60}\n")
                return True

        except Exception as e:
            print(f"❌ Error sending email: {e}")
            raise

    def send_welcome_email(self, to_email: str, name: str) -> bool:
        """Send welcome email after successful registration"""
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = 'Welcome to Medibot!'
            msg['From'] = self.from_email
            msg['To'] = to_email

            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        line-height: 1.6;
                        color: #333;
                        max-width: 600px;
                        margin: 0 auto;
                        padding: 20px;
                    }}
                    .header {{
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 30px;
                        text-align: center;
                        border-radius: 10px 10px 0 0;
                    }}
                    .content {{
                        background: #f9f9f9;
                        padding: 30px;
                        border: 1px solid #ddd;
                    }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>🤖 Welcome to MEDIBOT!</h1>
                </div>
                <div class="content">
                    <h2>Hello {name}!</h2>
                    <p>Welcome to the Medibot Medication Delivery Rover System. Your account has been successfully verified.</p>
                    <p>You can now log in and start using our platform to manage medication deliveries.</p>
                    <p>If you have any questions, feel free to contact our support team.</p>
                </div>
            </body>
            </html>
            """

            part = MIMEText(html_content, 'html')
            msg.attach(part)

            if self.smtp_password:
                with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                    server.starttls()
                    server.login(self.smtp_user, self.smtp_password)
                    server.send_message(msg)
            
            print(f"Welcome email sent to {to_email}")
            return True

        except Exception as e:
            print(f"Error sending welcome email: {e}")
            return False
