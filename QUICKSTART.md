# 🚀 Quick Start Guide

Get Medibot up and running in 5 minutes!

## Prerequisites

- Python 3.8+
- Modern web browser
- Internet connection (for Socket.IO CDN)

## Step 1: Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt
```

## Step 2: Start the System

### Option A: Using Startup Script (Recommended)

```bash
chmod +x start.sh
./start.sh
```

### Option B: Manual Start

```bash
cd backend
python app.py
```

## Step 3: Open Web Interface

Open your browser and navigate to:

```
http://localhost:5000
```

## Step 4: Explore the Dashboard

You should see:

✅ **Connection Panel** - Configure rover IP address
✅ **Camera Feed** - Will show live feed when connected
✅ **Navigation Map** - Interactive map with 26 keyframes
✅ **Rover Controls** - Joystick for manual control
✅ **Patients Panel** - 5 sample patients pre-loaded
✅ **System Monitor** - Real-time statistics

## Step 5: Test Without Hardware

Even without physical rover hardware, you can:

1. **View the map** - See all 26 keyframes and feature points
2. **Manage patients** - Add, edit, delete patient records
3. **Select keyframes** - Click on map to select destinations
4. **Explore the UI** - Test all responsive layouts

## Step 6: Connect to Rover (Optional)

If you have rover hardware:

1. Start rover control server:
   ```bash
   # On the rover (Jetson Nano, etc.)
   chmod +x start_rover.sh
   ./start_rover.sh
   ```

2. In the web interface:
   - Enter rover IP address (e.g., `192.168.1.100`)
   - Click **Connect**
   - Wait for confirmation

3. Once connected:
   - ✅ Camera feed will appear
   - ✅ Control buttons will activate
   - ✅ You can start navigation

## Testing Features

### Test Map Visualization

1. Look at the **Navigation Map** panel
2. You should see:
   - White background
   - Black feature points
   - Green circles (keyframes) labeled KF0-KF25
   - Grid lines

3. Try map controls:
   - Click **+** to zoom in
   - Click **−** to zoom out
   - Click **Reset** to reset view
   - Click on any keyframe to select it

### Test Patient Management

1. Go to **Patients** panel
2. You should see 5 sample patients:
   - John Smith (Room A101) → Keyframe 5
   - Mary Johnson (Room A102) → Keyframe 10
   - Robert Davis (Room B201) → Keyframe 15
   - Patricia Wilson (Room B202) → Keyframe 20
   - Michael Brown (Room C301) → Keyframe 25

3. Try patient operations:
   - Click **+ Add Patient** to add new patient
   - Click **✏️ Edit** to modify patient info
   - Click **🗑️ Delete** to remove patient
   - Click **🚀 Deliver** to start delivery (requires rover connection)

### Test System Monitor

1. Check the **System Monitor** panel
2. You should see:
   - Total deliveries count
   - Number of active patients
   - Current keyframe
   - System uptime

3. Activity log shows:
   - System initialization
   - All user actions
   - Navigation events

## Keyboard Shortcuts

When rover is connected:

- `W` or `↑` - Move forward
- `S` or `↓` - Move backward
- `A` or `←` - Turn left
- `D` or `→` - Turn right
- `Space` - Emergency stop

## Common Tasks

### Add a New Patient

1. Click **+ Add Patient**
2. Fill in the form:
   - **Name**: Patient full name
   - **Room Number**: e.g., "A101"
   - **Medication**: e.g., "Aspirin 100mg"
   - **Schedule**: e.g., "3 times daily"
   - **Location**: Select keyframe (where patient is located)
   - **Notes**: Any additional information
3. Click **Save Patient**

### Start a Delivery

1. Ensure rover is connected (green status dot)
2. Find patient in the list
3. Click **🚀 Deliver** button
4. Confirm the delivery
5. Watch the rover navigate on the map!

### Navigate Manually

1. Connect to rover
2. Use joystick buttons or keyboard shortcuts
3. Watch camera feed for real-time view
4. Check activity log for command confirmation

### Select Goal Point on Map

1. Click on any keyframe on the map
2. A confirmation dialog will appear
3. Click **OK** to start navigation
4. Watch the path line on the map
5. Monitor progress in activity log

## Troubleshooting

### Backend won't start

```bash
# Check Python version
python3 --version  # Should be 3.8+

# Reinstall dependencies
pip install -r requirements.txt

# Check port availability
lsof -i :5000  # Should be empty
```

### Can't see the map

1. Check browser console (F12) for errors
2. Verify keyframes exist: `ls data/keyframes/`
3. Try refreshing the page (Ctrl+R)
4. Clear browser cache

### Rover won't connect

1. Verify rover IP address is correct
2. Ping the rover: `ping 192.168.1.100`
3. Check rover server is running on port 8080
4. Verify firewall settings

## Next Steps

Once you're comfortable with the basics:

1. **Build your own map** - Record a video and generate new keyframes
2. **Customize patients** - Add your own patient database
3. **Integrate face recognition** - Add patient identification
4. **Connect LLM** - Add patient interaction capabilities
5. **Deploy to production** - Set up on hospital network

## Getting Help

- Check the full [README.md](README.md) for detailed documentation
- Review [API Documentation](README.md#api-documentation)
- Check the [Troubleshooting](README.md#troubleshooting) section
- Open an issue on GitHub

---

**Happy navigating! 🤖**
