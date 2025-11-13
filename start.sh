#!/bin/bash

# Medibot Startup Script
# Starts the complete Medication Delivery Rover system

echo "=========================================="
echo "  MEDIBOT - Medication Delivery Rover"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    exit 1
fi

echo "✓ Python found: $(python3 --version)"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo ""
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "📥 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo ""
echo "=========================================="
echo "  Starting Backend Server"
echo "=========================================="
echo ""
echo "🚀 Backend API: http://localhost:5000"
echo "🌐 Web Interface: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the backend server
cd backend
python app.py
