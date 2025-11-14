# Email Configuration for Medibot
# Copy this file to email_config.py and add your actual email credentials

# Gmail Configuration (Recommended)
# To use Gmail:
# 1. Go to https://myaccount.google.com/security
# 2. Enable 2-Step Verification
# 3. Generate an App Password: https://myaccount.google.com/apppasswords
# 4. Use the generated 16-character password below

SMTP_HOST = 'smtp.gmail.com'
SMTP_PORT = 587
SMTP_USER = 'your-email@gmail.com'  # Replace with your Gmail address
SMTP_PASSWORD = 'your-app-password'  # Replace with your App Password (16 characters)

# Alternative Email Providers:

# Outlook/Hotmail
# SMTP_HOST = 'smtp-mail.outlook.com'
# SMTP_PORT = 587
# SMTP_USER = 'your-email@outlook.com'
# SMTP_PASSWORD = 'your-password'

# Yahoo Mail
# SMTP_HOST = 'smtp.mail.yahoo.com'
# SMTP_PORT = 587
# SMTP_USER = 'your-email@yahoo.com'
# SMTP_PASSWORD = 'your-app-password'

# Custom SMTP Server
# SMTP_HOST = 'smtp.yourserver.com'
# SMTP_PORT = 587
# SMTP_USER = 'your-email@yourserver.com'
# SMTP_PASSWORD = 'your-password'

# Email Display Settings
FROM_NAME = 'Medibot System'
FROM_EMAIL = SMTP_USER
