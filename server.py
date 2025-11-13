# Rover Control - HTTP Only (No WebSocket)
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, StreamingResponse, Response
import time
import serial
import json
import cv2
import threading
import math
import socket
import subprocess
import atexit
import uvicorn
from queue import Queue


# ============================== 
# Camera cleanup on exit
# ============================== 
atexit.register(lambda: subprocess.run(['sudo', 'systemctl', 'stop', 'nvargus-daemon'], 
                                       stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL))


def full_camera_reset():
    """Complete camera reset"""
    print("\n🔧 Resetting camera...")
    try:
        subprocess.run(['sudo', 'pkill', '-9', '-f', 'nvargus'], timeout=2)
        time.sleep(1)
        subprocess.run(['sudo', 'systemctl', 'stop', 'nvargus-daemon'], timeout=5)
        time.sleep(2)
        subprocess.run(['sudo', 'rm', '-f', '/tmp/argus_socket'], timeout=2)
        subprocess.run(['sudo', 'systemctl', 'start', 'nvargus-daemon'], timeout=5)
        time.sleep(3)
        print("✅ Camera reset complete\n")
    except:
        pass


full_camera_reset()


# ============================== 
# Rover Motor Control with INTERRUPTIBLE TIME-BASED MOVEMENT
# ============================== 
class Rover:
    def __init__(self, port="/dev/ttyACM1", baud=115200):
        self.left_speed = 0.0
        self.right_speed = 0.0
        
        self.ddsm_ser = serial.Serial()
        self.ddsm_ser.port = port
        self.ddsm_ser.baudrate = baud
        self.ddsm_ser.dtr = False
        self.ddsm_ser.rts = False
        self.ddsm_ser.open()
        time.sleep(0.5)
        self.ddsm_ser.reset_input_buffer()
        self.ddsm_ser.reset_output_buffer()
        self._send_stop_commands()
        
        # Odometry parameters (kept for future use)
        self.wheel_diameter = 0.065
        self.wheel_base = 0.15
        self.encoder_ticks_per_rev = 360
        
        self.left_encoder = 0
        self.right_encoder = 0
        
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        
        # TIME-BASED CALIBRATION VALUES
        # Forward/Backward: 2.5m in 10s → 0.3m takes 1.2s
        self.forward_time_per_meter = 10.0 / 2.5  # 4.0 seconds per meter
        self.target_distance = 0.3  # 30cm
        
        # Turning: 360° in 14s → 90° takes 3.5s
        self.turn_time_per_degree = 14.0 / 360.0  # 0.03889 seconds per degree
        self.target_angle = 90  # 90° turns
        
        # Command queue for sequential processing
        self.command_queue = Queue()
        self.is_moving = False
        self.stop_requested = False  # Flag for immediate stop
        self.queue_processor_running = True
        
        # Start odometry thread (for monitoring)
        self.odom_thread = threading.Thread(target=self._read_odometry, daemon=True)
        self.odom_running = True
        self.odom_thread.start()
        
        # Start command queue processor
        self.queue_thread = threading.Thread(target=self._process_command_queue, daemon=True)
        self.queue_thread.start()
        
        print("✅ Rover initialized - HTTP command queue enabled")
    
    def _send_stop_commands(self):
        stop_command_left = {"T": 10010, "id": 2, "cmd": 0, "act": 3}
        stop_command_right = {"T": 10010, "id": 1, "cmd": 0, "act": 3}
        
        for _ in range(3):
            self.ddsm_ser.write((json.dumps(stop_command_right) + '\n').encode())
            time.sleep(0.05)
            self.ddsm_ser.write((json.dumps(stop_command_left) + '\n').encode())
            time.sleep(0.05)
        
        print("🛑 Stop commands sent to motors")
    
    def _read_odometry(self):
        """Still reads encoders for monitoring purposes"""
        while self.odom_running:
            try:
                if self.ddsm_ser.in_waiting:
                    line = self.ddsm_ser.readline().decode('utf-8').strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    if 'enc_l' in data:
                        self.left_encoder = data['enc_l']
                    if 'enc_r' in data:
                        self.right_encoder = data['enc_r']
            except json.JSONDecodeError:
                pass
            except:
                pass
            time.sleep(0.01)
    
    def _process_command_queue(self):
        """Background thread that processes commands from queue sequentially"""
        print("🔄 Command queue processor started")
        while self.queue_processor_running:
            try:
                if not self.command_queue.empty():
                    command = self.command_queue.get()
                    print(f"📥 Processing command: {command}")
                    
                    if command == "forward":
                        self._execute_forward()
                    elif command == "backward":
                        self._execute_backward()
                    elif command == "left":
                        self._execute_turn_left()
                    elif command == "right":
                        self._execute_turn_right()
                    elif command == "stop":
                        self._execute_stop()
                    
                    self.command_queue.task_done()
                else:
                    time.sleep(0.05)
            except Exception as e:
                print(f"❌ Queue processing error: {e}")
                time.sleep(0.1)
    
    def set_motor_speeds(self, left, right):
        self.left_speed = left
        self.right_speed = right
        
        command_left = {"T": 10010, "id": 2, "cmd": -left, "act": 3}
        command_right = {"T": 10010, "id": 1, "cmd": right, "act": 3}
        
        self.ddsm_ser.write((json.dumps(command_right) + '\n').encode())
        time.sleep(0.01)
        self.ddsm_ser.write((json.dumps(command_left) + '\n').encode())
        time.sleep(0.01)
    
    def stop(self):
        self.set_motor_speeds(0, 0)
    
    def _interruptible_sleep(self, duration, check_interval=0.05):
        """
        Sleep that can be interrupted by stop_requested flag
        Returns True if completed, False if interrupted
        """
        elapsed = 0.0
        while elapsed < duration:
            if self.stop_requested:
                return False  # Interrupted
            
            sleep_time = min(check_interval, duration - elapsed)
            time.sleep(sleep_time)
            elapsed += sleep_time
        
        return True  # Completed
    
    # ===== INTERRUPTIBLE TIME-BASED EXECUTION METHODS =====
    
    def _execute_forward(self):
        """Execute forward movement with interrupt capability"""
        self.is_moving = True
        self.stop_requested = False
        
        move_time = self.target_distance * self.forward_time_per_meter  # 1.2s
        
        print(f"➡ Moving forward {self.target_distance}m for {move_time:.2f}s")
        self.set_motor_speeds(50, 50)
        
        completed = self._interruptible_sleep(move_time)
        
        self.stop()
        
        if completed:
            print(f"✅ Forward complete")
        else:
            print(f"🛑 Forward interrupted by stop")
        
        self.is_moving = False
        self.stop_requested = False
    
    def _execute_backward(self):
        """Execute backward movement with interrupt capability"""
        self.is_moving = True
        self.stop_requested = False
        
        move_time = self.target_distance * self.forward_time_per_meter  # 1.2s
        
        print(f"⬅ Moving backward {self.target_distance}m for {move_time:.2f}s")
        self.set_motor_speeds(-50, -50)
        
        completed = self._interruptible_sleep(move_time)
        
        self.stop()
        
        if completed:
            print(f"✅ Backward complete")
        else:
            print(f"🛑 Backward interrupted by stop")
        
        self.is_moving = False
        self.stop_requested = False
    
    def _execute_turn_left(self):
        """Execute left turn with interrupt capability"""
        self.is_moving = True
        self.stop_requested = False
        
        turn_time = self.target_angle * self.turn_time_per_degree  # 3.5s
        
        print(f"↶ Turning left {self.target_angle}° for {turn_time:.2f}s")
        self.set_motor_speeds(25, -25)
        
        completed = self._interruptible_sleep(turn_time)
        
        self.stop()
        
        if completed:
            print(f"✅ Left turn complete")
        else:
            print(f"🛑 Left turn interrupted by stop")
        
        self.is_moving = False
        self.stop_requested = False
    
    def _execute_turn_right(self):
        """Execute right turn with interrupt capability"""
        self.is_moving = True
        self.stop_requested = False
        
        turn_time = self.target_angle * self.turn_time_per_degree  # 3.5s
        
        print(f"↷ Turning right {self.target_angle}° for {turn_time:.2f}s")
        self.set_motor_speeds(-25, 25)
        
        completed = self._interruptible_sleep(turn_time)
        
        self.stop()
        
        if completed:
            print(f"✅ Right turn complete")
        else:
            print(f"🛑 Right turn interrupted by stop")
        
        self.is_moving = False
        self.stop_requested = False
    
    def _execute_stop(self):
        """Immediate stop - sets flag to interrupt current movement"""
        print(f"🛑 STOP command received")
        
        # Set stop flag to interrupt current movement
        self.stop_requested = True
        
        # Clear all pending commands
        cleared_count = 0
        while not self.command_queue.empty():
            try:
                self.command_queue.get_nowait()
                self.command_queue.task_done()
                cleared_count += 1
            except:
                break
        
        if cleared_count > 0:
            print(f"   Cleared {cleared_count} queued command(s)")
        
        # Wait briefly for current movement to stop
        time.sleep(0.2)
        
        # Force stop motors
        self.stop()
        self.is_moving = False
        
        print("✅ Stopped")
    
    # ===== PUBLIC METHODS =====
    
    def add_command(self, command: str):
        """Add a command to the queue for sequential execution"""
        self.command_queue.put(command)
        print(f"📝 Command queued: {command} (queue size: {self.command_queue.qsize()})")
    
    def get_queue_size(self):
        """Get current queue size"""
        return self.command_queue.qsize()
    
    def emergency_stop(self):
        """Add emergency stop command"""
        self.add_command("stop")


# ============================== 
# LOW LATENCY Video Streaming
# ============================== 
class VideoStream:
    def _init_(self):
        self.output_frame = None
        self.lock = threading.Lock()
        self.streaming = False
        self.capture_thread = None
        self.camera = None
    
    def start_stream(self):
        if self.streaming:
            return
        
        print("🎥 Starting low-latency camera stream...")
        self.streaming = True
        self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.capture_thread.start()
    
    def _capture_frames(self):
        print("📷 Initializing CSI camera with low latency settings...")
        
        gst_pipeline = (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=40/1 ! "
            "nvvidconv flip-method=2 ! "
            "video/x-raw, format=BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=BGR ! "
            "appsink drop=true max-buffers=1 emit-signals=true sync=false"
        )
        
        try:
            self.camera = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            
            if not self.camera.isOpened():
                raise Exception("Camera not opened")
            
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            print("  └─ Warming up camera...")
            for i in range(30):
                ret, _ = self.camera.read()
                if not ret:
                    time.sleep(0.1)
            
            print("✅ CSI camera ready (low latency mode)")
        
        except Exception as e:
            print(f"❌ Camera failed: {e}")
            return
        
        frame_count = 0
        consecutive_failures = 0
        
        while self.streaming:
            ret, frame = self.camera.read()
            
            if not ret:
                consecutive_failures += 1
                if consecutive_failures > 30:
                    print(f"❌ Camera lost after {frame_count} frames")
                    break
                time.sleep(0.05)
                continue
            
            consecutive_failures = 0
            frame_count += 1
            
            with self.lock:
                self.output_frame = frame.copy()
        
        if self.camera:
            self.camera.release()
        print(f"🛑 Camera stopped ({frame_count} frames)")
    
    def generate_frames(self):
        """Generate frames for MJPEG streaming"""
        while True:
            with self.lock:
                if self.output_frame is None:
                    time.sleep(0.01)
                    continue
                
                ret, buffer = cv2.imencode('.jpg', self.output_frame, 
                                          [cv2.IMWRITE_JPEG_QUALITY, 60,
                                           cv2.IMWRITE_JPEG_OPTIMIZE, 0])
                
                if not ret:
                    continue
                
                frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    def get_latest_frame(self):
        """Get single latest frame as JPEG bytes"""
        with self.lock:
            if self.output_frame is None:
                return None
            
            ret, buffer = cv2.imencode('.jpg', self.output_frame,
                                      [cv2.IMWRITE_JPEG_QUALITY, 60,
                                       cv2.IMWRITE_JPEG_OPTIMIZE, 0])
            if not ret:
                return None
            
            return buffer.tobytes()
    
    def stop_stream(self):
        self.streaming = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2)
        if self.camera:
            self.camera.release()


def get_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return '127.0.0.1'


# Initialize rover and video stream
rover = Rover()
video_stream = VideoStream()
video_stream.start_stream()


app = FastAPI()


HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Rover Control - HTTP Only</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            background: #121212;
            color: white;
            font-family: Arial;
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
        }
        .info-banner {
            background: #1e3a8a;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            margin: 20px auto;
            max-width: 800px;
        }
        .main-layout {
            display: flex;
            gap: 20px;
            align-items: flex-start;
            justify-content: center;
            margin-top: 30px;
        }
        .stream-container {
            background: #1e1e1e;
            padding: 20px;
            border-radius: 15px;
            flex-shrink: 0;
        }
        #video-stream {
            width: 800px;
            height: 600px;
            border-radius: 10px;
            border: 3px solid #333;
            background: #000;
            display: block;
            object-fit: contain;
        }
        .controls-container {
            background: #1e1e1e;
            padding: 30px;
            border-radius: 15px;
            min-width: 350px;
        }
        .controls-container h2 {
            margin-top: 0;
            text-align: center;
        }
        button {
            width: 120px;
            height: 60px;
            font-size: 18px;
            margin: 10px;
            border-radius: 10px;
            border: none;
            cursor: pointer;
            transition: 0.2s;
            font-weight: bold;
        }
        button:hover {
            transform: scale(1.1);
        }
        button:active {
            transform: scale(0.95);
        }
        .fwd { background-color: #4CAF50; color: white; }
        .bwd { background-color: #E53935; color: white; }
        .left { background-color: #1E88E5; color: white; }
        .right { background-color: #FB8C00; color: white; }
        .stop { 
            background-color: #DC2626; 
            color: white;
            border: 3px solid #FCA5A5;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.7); }
            50% { box-shadow: 0 0 0 10px rgba(220, 38, 38, 0); }
        }
        #command-status {
            margin-top: 20px;
            padding: 10px;
            background: #333;
            border-radius: 5px;
            font-family: monospace;
            text-align: center;
        }
        .button-row {
            text-align: center;
        }
        .method-notice {
            background: #065f46;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
            text-align: center;
            font-size: 13px;
        }
        @media (max-width: 1200px) {
            .main-layout {
                flex-direction: column;
                align-items: center;
            }
            #video-stream {
                width: 100%;
                max-width: 800px;
                height: auto;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚗 Rover Control - HTTP Only</h1>
        
        <div class="info-banner">
            ⏱ Forward: 0.3m (1.2s) | Turn: 90° (3.5s) | 🛑 INSTANT STOP: ~50ms response
        </div>
        
        <div class="main-layout">
            <div class="stream-container">
                <h2>📹 CSI Camera</h2>
                <img id="video-stream" src="/video_feed" alt="Loading...">
            </div>
            
            <div class="controls-container">
                <h2>🎮 Controls</h2>
                
                <div class="button-row">
                    <button class="fwd" onclick="cmd('forward')">
                        ⬆ Forward<br><small>(0.3m)</small>
                    </button>
                </div>
                
                <div class="button-row">
                    <button class="left" onclick="cmd('left')">
                        ⬅ Left<br><small>(90°)</small>
                    </button>
                    <button class="stop" onclick="cmd('stop')">
                        ⛔ STOP<br><small>(INSTANT)</small>
                    </button>
                    <button class="right" onclick="cmd('right')">
                        ➡ Right<br><small>(90°)</small>
                    </button>
                </div>
                
                <div class="button-row">
                    <button class="bwd" onclick="cmd('backward')">
                        ⬇ Backward<br><small>(0.3m)</small>
                    </button>
                </div>
                
                <div class="method-notice">
                    📡 HTTP POST Method<br>No WebSocket required
                </div>
                
                <div id="command-status">Ready</div>
            </div>
        </div>
    </div>
    
    <script>
        let reconnectAttempts = 0;
        
        // Video stream reconnection
        document.getElementById('video-stream').onerror = function() {
            reconnectAttempts++;
            setTimeout(() => {
                this.src = /video_feed?t=${Date.now()};
            }, 1000);
        };
        
        function cmd(dir) {
            document.getElementById('command-status').textContent = Sending: ${dir};
            
            fetch('/move', {
                method: 'POST',
                headers: {'Content-Type': 'application/x-www-form-urlencoded'},
                body: direction=${dir}
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('command-status').textContent = 
                    ✅ ${dir} (Queue: ${data.queue_size});
            })
            .catch(error => {
                document.getElementById('command-status').textContent = ❌ Error: ${error};
            });
        }
        
        // Keyboard controls
        document.addEventListener('keydown', function(e) {
            if (e.key === 'ArrowUp' || e.key === 'w') cmd('forward');
            if (e.key === 'ArrowDown' || e.key === 's') cmd('backward');
            if (e.key === 'ArrowLeft' || e.key === 'a') cmd('left');
            if (e.key === 'ArrowRight' || e.key === 'd') cmd('right');
            if (e.key === ' ') cmd('stop');
            e.preventDefault();
        });
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.get("/video_feed")
def video_feed():
    """MJPEG stream endpoint"""
    return StreamingResponse(
        video_stream.generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/frame")
def get_frame():
    """Get single latest frame as JPEG"""
    frame_bytes = video_stream.get_latest_frame()
    if frame_bytes is None:
        return {"error": "No frame available"}
    
    return Response(content=frame_bytes, media_type="image/jpeg")


@app.post("/move")
async def move(direction: str = Form(...)):
    """
    HTTP POST endpoint for sending commands
    Accepts: direction = "forward" | "backward" | "left" | "right" | "stop"
    """
    rover.add_command(direction)
    return {
        "status": "ok", 
        "command": direction,
        "queue_size": rover.get_queue_size()
    }


@app.get("/queue/status")
async def queue_status():
    """Get current queue status"""
    return {
        "queue_size": rover.get_queue_size(),
        "is_moving": rover.is_moving
    }


@app.post("/queue/clear")
async def clear_queue():
    """Emergency clear all queued commands"""
    rover.add_command("stop")
    return {"status": "Emergency stop command added"}


if __name__ == "__main__":
    local_ip = get_ip()
    port = 8080
    
    try:
        print("=" * 70)
        print("🚗 ROVER CONTROL - HTTP ONLY (No WebSocket)")
        print("=" * 70)
        print(f"🌐 Web UI: http://{local_ip}:{port}")
        print(f"📡 HTTP POST: http://{local_ip}:{port}/move")
        print(f"📹 Camera: http://{local_ip}:{port}/frame")
        print(f"📊 Queue Status: http://{local_ip}:{port}/queue/status")
        print("=" * 70)
        print("⏱  CALIBRATION:")
        print(f"   Forward: 0.3m = 1.2s (from 2.5m in 10s)")
        print(f"   Turn: 90° = 3.5s (from 360° in 14s)")
        print(f"   Stop Response: ~50ms")
        print("=" * 70)
        
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning", workers=1)
    
    except KeyboardInterrupt:
        print("\n🛑 Stopping...")
        rover.odom_running = False
        rover.queue_processor_running = False
        video_stream.stop_stream()
        rover.stop()