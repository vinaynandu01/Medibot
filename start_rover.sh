#!/bin/bash

# Rover Control Server Startup Script
# Run this on the rover hardware (Jetson Nano, etc.)

echo "=========================================="
echo "  MEDIBOT - Rover Control Server"
echo "=========================================="
echo ""

# Default rover IP (will bind to all interfaces)
ROVER_IP=${ROVER_IP:-"0.0.0.0"}
ROVER_PORT=${ROVER_PORT:-8080}

echo "📡 Starting rover control server..."
echo "   IP: $ROVER_IP"
echo "   Port: $ROVER_PORT"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    exit 1
fi

# Install dependencies if needed
pip3 install -q fastapi uvicorn pyserial

# Start the rover server
cd rover/control
python3 server.py --host $ROVER_IP --port $ROVER_PORT
