import cv2
import requests
import numpy as np
import time
from datetime import datetime
import threading
import queue

class SLAMVideoRecorder:
    def __init__(self, server_url="http://10.226.57.127:8080"):
        self.server_url = server_url
        self.video_feed_url = f"{server_url}/video_feed"
        
        # Queue to hold frames - unlimited size to ensure no frames dropped
        self.frame_queue = queue.Queue(maxsize=0)
        
        # Recording state
        self.is_recording = False
        self.video_writer = None
        self.recording_thread = None
        self.capture_thread = None
        
        # Display
        self.current_display_frame = None
        self.display_lock = threading.Lock()
        
        # Stats
        self.frames_captured = 0
        self.frames_written = 0
        self.start_time = None
        
    def start_recording(self, output_filename=None, fps=30, resolution=(1280, 720)):
        """Start recording frames to video file"""
        if self.is_recording:
            print("⚠️ Already recording!")
            return False
        
        # Generate filename if not provided
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"slam_map_{timestamp}.mp4"
        
        print(f"🎬 Starting recording: {output_filename}")
        print(f"   Resolution: {resolution[0]}x{resolution[1]} @ {fps}fps")
        
        # Initialize video writer with H264 codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'avc1' for better compatibility
        self.video_writer = cv2.VideoWriter(
            output_filename,
            fourcc,
            fps,
            resolution
        )
        
        if not self.video_writer.isOpened():
            print("❌ Failed to initialize video writer!")
            return False
        
        # Reset stats
        self.frames_captured = 0
        self.frames_written = 0
        self.start_time = time.time()
        self.is_recording = True
        
        # Clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                break
        
        # Start capture thread (high priority)
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        # Start writing thread
        self.recording_thread = threading.Thread(target=self._write_loop, daemon=True)
        self.recording_thread.start()
        
        print("✅ Recording started")
        return True
    
    def _capture_loop(self):
        """Capture frames from MJPEG stream - NO frame skipping"""
        consecutive_failures = 0
        
        try:
            # Open MJPEG stream
            stream = requests.get(self.video_feed_url, stream=True, timeout=5)
            
            if stream.status_code != 200:
                print(f"❌ Failed to connect to video feed: {stream.status_code}")
                self.is_recording = False
                return
            
            print("✅ Connected to video stream")
            
            # Buffer for MJPEG parsing
            byte_buffer = b''
            
            for chunk in stream.iter_content(chunk_size=4096):
                if not self.is_recording:
                    break
                
                byte_buffer += chunk
                
                # Find JPEG boundaries
                while True:
                    # Look for JPEG start
                    start_idx = byte_buffer.find(b'\xff\xd8')
                    if start_idx == -1:
                        break
                    
                    # Look for JPEG end
                    end_idx = byte_buffer.find(b'\xff\xd9', start_idx + 2)
                    if end_idx == -1:
                        break
                    
                    # Extract complete JPEG
                    jpg = byte_buffer[start_idx:end_idx + 2]
                    byte_buffer = byte_buffer[end_idx + 2:]
                    
                    # Decode frame
                    img_array = np.frombuffer(jpg, dtype=np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        # Add to queue immediately
                        self.frame_queue.put((time.time(), frame))
                        self.frames_captured += 1
                        consecutive_failures = 0
                    
        except Exception as e:
            print(f"❌ Stream error: {e}")
            self.is_recording = False
    
    def _write_loop(self):
        """Write frames from queue to video file"""
        while self.is_recording or not self.frame_queue.empty():
            try:
                # Get frame from queue (with timeout to check recording status)
                timestamp, frame = self.frame_queue.get(timeout=1.0)
                
                # Write frame to video
                self.video_writer.write(frame)
                self.frames_written += 1
                
                # Update display frame
                with self.display_lock:
                    self.current_display_frame = frame.copy()
                
                # Print progress every 100 frames
                if self.frames_written % 100 == 0:
                    elapsed = time.time() - self.start_time
                    fps = self.frames_written / elapsed
                    queue_size = self.frame_queue.qsize()
                    print(f"📊 Frames: {self.frames_written} | FPS: {fps:.1f} | Queue: {queue_size}")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Write error: {e}")
                break
    
    def get_display_frame(self):
        """Get current frame for display"""
        with self.display_lock:
            if self.current_display_frame is not None:
                return self.current_display_frame.copy()
        return None
    
    def stop_recording(self):
        """Stop recording and save video file"""
        if not self.is_recording:
            print("⚠️ Not recording!")
            return False
        
        print("\n⏹️ Stopping recording...")
        self.is_recording = False
        
        # Wait for threads to finish
        if self.capture_thread:
            self.capture_thread.join(timeout=5)
        if self.recording_thread:
            self.recording_thread.join(timeout=10)
        
        # Release video writer
        if self.video_writer:
            self.video_writer.release()
        
        # Calculate stats
        elapsed = time.time() - self.start_time
        actual_fps = self.frames_written / elapsed if elapsed > 0 else 0
        
        print(f"✅ Recording complete!")
        print(f"   Frames captured: {self.frames_captured}")
        print(f"   Frames written: {self.frames_written}")
        print(f"   Duration: {elapsed:.1f}s")
        print(f"   Actual FPS: {actual_fps:.1f}")
        print(f"   Frame loss: {self.frames_captured - self.frames_written}")
        
        return True
    
    def get_stats(self):
        """Get current recording statistics"""
        if not self.is_recording:
            return None
        
        elapsed = time.time() - self.start_time
        return {
            "frames_captured": self.frames_captured,
            "frames_written": self.frames_written,
            "queue_size": self.frame_queue.qsize(),
            "elapsed_time": elapsed,
            "capture_fps": self.frames_captured / elapsed if elapsed > 0 else 0,
            "write_fps": self.frames_written / elapsed if elapsed > 0 else 0
        }


def main():
    """Main function with live preview window"""
    import sys
    
    # Get server URL from command line or use default
    server_url = sys.argv[1] if len(sys.argv) > 1 else "http://10.109.85.127:8080"
    
    recorder = SLAMVideoRecorder(server_url)
    
    print("=" * 60)
    print("🎥 SLAM Video Recorder with Live Preview")
    print("=" * 60)
    print(f"Server: {server_url}")
    print("\nStarting recording...")
    print("Press 'S' in the video window to STOP and SAVE")
    print("=" * 60)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"slam_map_{timestamp}.mp4"
    
    # Start recording
    if not recorder.start_recording(output_filename, fps=40):
        print("❌ Failed to start recording!")
        return
    
    # Create window
    window_name = "SLAM Recording - Press 'S' to Stop"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    
    print("\n🟢 RECORDING - Window opened")
    
    try:
        while recorder.is_recording:
            # Get current frame
            frame = recorder.get_display_frame()
            
            if frame is not None:
                # Add recording indicator
                display_frame = frame.copy()
                
                # Red circle (recording indicator)
                cv2.circle(display_frame, (30, 30), 15, (0, 0, 255), -1)
                
                # Add text overlay
                elapsed = time.time() - recorder.start_time
                text = f"REC {elapsed:.1f}s | Frames: {recorder.frames_written}"
                cv2.putText(display_frame, text, (60, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Show FPS
                stats = recorder.get_stats()
                if stats:
                    fps_text = f"FPS: {stats['write_fps']:.1f}"
                    cv2.putText(display_frame, fps_text, (60, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show instructions
                cv2.putText(display_frame, "Press 'S' to Stop", (60, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow(window_name, display_frame)
            
            # Check for keypress
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s') or key == ord('S'):
                print("\n⏹️ Stop key pressed!")
                break
            elif key == 27:  # ESC key
                print("\n⏹️ ESC pressed!")
                break
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted!")
    
    finally:
        # Stop recording and cleanup
        recorder.stop_recording()
        cv2.destroyAllWindows()
        print(f"\n✅ Video saved: {output_filename}")


if __name__ == "__main__":
    main()