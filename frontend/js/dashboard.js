/*
 * Dashboard Controller
 * Main controller for rover monitoring and control
 */

class Dashboard {
  constructor() {
    this.isConnected = false;
    this.roverIP = CONFIG.DEFAULT_ROVER_IP;
    this.socket = null;
    this.updateInterval = null;
    this.startTime = Date.now();

    this.setupEventListeners();
    this.setupWebSocket();
    this.startUptimeCounter();
  }

  setupEventListeners() {
    // Connection buttons
    document.getElementById("connectBtn").addEventListener("click", () => {
      this.connectRover();
    });

    document.getElementById("disconnectBtn").addEventListener("click", () => {
      this.disconnectRover();
    });

    // Control buttons
    document.getElementById("forwardBtn").addEventListener("click", () => {
      this.moveRover("forward");
    });

    document.getElementById("backwardBtn").addEventListener("click", () => {
      this.moveRover("backward");
    });

    document.getElementById("leftBtn").addEventListener("click", () => {
      this.moveRover("left");
    });

    document.getElementById("rightBtn").addEventListener("click", () => {
      this.moveRover("right");
    });

    document.getElementById("stopBtn").addEventListener("click", () => {
      this.stopRover();
    });

    // Snapshot button
    document.getElementById("snapshotBtn").addEventListener("click", () => {
      this.takeSnapshot();
    });

    // Map controls
    document.getElementById("zoomInBtn").addEventListener("click", () => {
      if (mapVisualizer) mapVisualizer.zoomIn();
    });

    document.getElementById("zoomOutBtn").addEventListener("click", () => {
      if (mapVisualizer) mapVisualizer.zoomOut();
    });

    document.getElementById("resetViewBtn").addEventListener("click", () => {
      if (mapVisualizer) mapVisualizer.resetView();
    });

    // Keyboard controls
    document.addEventListener("keydown", (e) => {
      if (this.isConnected && !this.isTyping()) {
        switch (e.key) {
          case "ArrowUp":
          case "w":
          case "W":
            e.preventDefault();
            this.moveRover("forward");
            break;
          case "ArrowDown":
          case "s":
          case "S":
            e.preventDefault();
            this.moveRover("backward");
            break;
          case "ArrowLeft":
          case "a":
          case "A":
            e.preventDefault();
            this.moveRover("left");
            break;
          case "ArrowRight":
          case "d":
          case "D":
            e.preventDefault();
            this.moveRover("right");
            break;
          case " ":
            e.preventDefault();
            this.stopRover();
            break;
        }
      }
    });

    // Listen for keyframe selection
    window.addEventListener("keyframe-selected", (e) => {
      this.handleKeyframeSelected(e.detail.keyframeId);
    });
  }

  isTyping() {
    const activeElement = document.activeElement;
    return (
      activeElement.tagName === "INPUT" ||
      activeElement.tagName === "TEXTAREA" ||
      activeElement.isContentEditable
    );
  }

  setupWebSocket() {
    try {
      this.socket = io(CONFIG.SOCKET_URL);

      this.socket.on("connect", () => {
        console.log("WebSocket connected");
        this.logActivity("WebSocket connection established");
      });

      this.socket.on("disconnect", () => {
        console.log("WebSocket disconnected");
        this.logActivity("WebSocket connection lost");
      });

      this.socket.on("rover_connected", (data) => {
        console.log("Rover connected:", data);
        this.logActivity(`Rover connected at ${data.ip_address}`);
      });

      this.socket.on("rover_disconnected", () => {
        console.log("Rover disconnected");
        this.updateConnectionStatus(false);
      });

      this.socket.on("rover_state", (state) => {
        this.updateRoverState(state);
      });

      this.socket.on("rover_moved", (data) => {
        this.logActivity(`Rover moved ${data.direction}`);
      });

      this.socket.on("rover_stopped", () => {
        this.logActivity("Rover stopped");
      });

      this.socket.on("navigation_started", (data) => {
        this.logActivity(`Navigation started to keyframe ${data.target}`);
      });

      this.socket.on("delivery_started", (data) => {
        this.logActivity(`Delivery started to ${data.patient_name}`);
      });
    } catch (error) {
      console.error("WebSocket error:", error);
    }
  }

  async connectRover() {
    const ipInput = document.getElementById("roverIP");
    this.roverIP = ipInput.value || CONFIG.DEFAULT_ROVER_IP;

    const connectBtn = document.getElementById("connectBtn");
    const messageDiv = document.getElementById("connectionMessage");

    connectBtn.disabled = true;
    connectBtn.textContent = "Connecting...";

    // Show detailed connection attempt
    messageDiv.innerHTML = `
            <div style="padding: 12px; background: #e3f2fd; border-left: 4px solid #2196f3; border-radius: 4px;">
                <div style="font-weight: 600; margin-bottom: 8px;">🔌 Connecting to ${this.roverIP}...</div>
                <div style="font-size: 0.85rem; color: #555;">
                    • Testing connection on port 8080<br>
                    • Waiting for rover response (max 10s)<br>
                    • Make sure rover server is running
                </div>
            </div>
        `;

    try {
      const response = await api.connectRover(this.roverIP);

      if (response.success) {
        this.isConnected = true;
        this.updateConnectionStatus(true);
        this.showConnectionMessage(
          `✅ Connected successfully to ${this.roverIP}${
            response.port ? ":" + response.port : ""
          }`,
          "success"
        );
        this.startCameraFeed();
        this.logActivity(`Connected to rover at ${this.roverIP}`);
      } else {
        throw new Error(response.message || "Connection failed");
      }
    } catch (error) {
      console.error("Connection error:", error);
      const errorMsg =
        error.response?.data?.message || error.message || "Unknown error";
      this.showConnectionMessage(
        `❌ Connection Failed to ${this.roverIP}<br><br>` +
          `<strong>Error:</strong> ${errorMsg}<br><br>` +
          `<strong>Troubleshooting:</strong><br>` +
          `• Check rover is powered on<br>` +
          `• Verify IP address: ${this.roverIP}<br>` +
          `• Ensure both devices on same network<br>` +
          `• Confirm rover server is running (check rover terminal)<br>` +
          `• Try pinging: <code>ping ${this.roverIP}</code>`,
        "error"
      );
    } finally {
      connectBtn.disabled = false;
      connectBtn.textContent = "Connect";
    }
  }

  async disconnectRover() {
    try {
      await api.disconnectRover();
      this.isConnected = false;
      this.updateConnectionStatus(false);
      this.stopCameraFeed();
      this.showConnectionMessage("Disconnected", "success");
      this.logActivity("Disconnected from rover");
    } catch (error) {
      console.error("Disconnect error:", error);
    }
  }

  updateConnectionStatus(connected) {
    this.isConnected = connected;

    // Update status indicator
    const statusDot = document.getElementById("statusDot");
    const statusText = document.getElementById("statusText");

    if (connected) {
      statusDot.classList.add("connected");
      statusDot.classList.remove("disconnected");
      statusText.textContent = "Connected";
    } else {
      statusDot.classList.add("disconnected");
      statusDot.classList.remove("connected");
      statusText.textContent = "Disconnected";
    }

    // Update button states
    document.getElementById("connectBtn").disabled = connected;
    document.getElementById("disconnectBtn").disabled = !connected;

    // Update control buttons
    const controlButtons = [
      "forwardBtn",
      "backwardBtn",
      "leftBtn",
      "rightBtn",
      "stopBtn",
    ];
    controlButtons.forEach((id) => {
      document.getElementById(id).disabled = !connected;
    });

    document.getElementById("snapshotBtn").disabled = !connected;
  }

  showConnectionMessage(message, type) {
    const messageDiv = document.getElementById("connectionMessage");
    messageDiv.textContent = message;
    messageDiv.className = `connection-status ${type}`;

    setTimeout(() => {
      messageDiv.textContent = "";
      messageDiv.className = "connection-status";
    }, 3000);
  }

  startCameraFeed() {
    const cameraFeed = document.getElementById("cameraFeed");
    const noFeed = document.getElementById("noFeed");

    cameraFeed.src = api.getCameraFeedURL() + "?t=" + Date.now();
    cameraFeed.style.display = "block";
    noFeed.style.display = "none";

    this.logActivity("Camera feed started");
  }

  stopCameraFeed() {
    const cameraFeed = document.getElementById("cameraFeed");
    const noFeed = document.getElementById("noFeed");

    cameraFeed.src = "";
    cameraFeed.style.display = "none";
    noFeed.style.display = "flex";

    this.logActivity("Camera feed stopped");
  }

  async moveRover(direction) {
    if (!this.isConnected) return;

    try {
      await api.moveRover(direction);
      this.logActivity(`Moving ${direction}`);
    } catch (error) {
      console.error("Move error:", error);
      this.logActivity(`Error: Failed to move ${direction}`, "error");
    }
  }

  async stopRover() {
    if (!this.isConnected) return;

    try {
      await api.stopRover();
      this.logActivity("Emergency stop");
    } catch (error) {
      console.error("Stop error:", error);
    }
  }

  async takeSnapshot() {
    if (!this.isConnected) return;

    try {
      const snapshotURL = api.getSnapshotURL() + "?t=" + Date.now();

      // Download snapshot
      const a = document.createElement("a");
      a.href = snapshotURL;
      a.download = `medibot_snapshot_${Date.now()}.jpg`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);

      this.logActivity("Snapshot saved");
    } catch (error) {
      console.error("Snapshot error:", error);
    }
  }

  async handleKeyframeSelected(keyframeId) {
    if (!this.isConnected) {
      alert("Please connect to rover first");
      if (mapVisualizer) mapVisualizer.clearSelection();
      return;
    }

    const confirmed = confirm(`Navigate to Keyframe #${keyframeId}?`);

    if (confirmed) {
      try {
        await api.navigateToKeyframe(keyframeId);
        this.logActivity(`Navigating to Keyframe #${keyframeId}`);
      } catch (error) {
        console.error("Navigation error:", error);
        this.logActivity(`Error: Navigation failed`, "error");
        alert("Navigation failed: " + error.message);
      }
    } else {
      if (mapVisualizer) mapVisualizer.clearSelection();
    }
  }

  updateRoverState(state) {
    // Update battery
    const batteryLevel = document.getElementById("batteryLevel");
    const batteryText = document.getElementById("batteryText");
    batteryLevel.style.width = state.battery + "%";
    batteryText.textContent = state.battery + "%";

    if (state.battery < 20) {
      batteryLevel.classList.add("low");
    } else {
      batteryLevel.classList.remove("low");
    }

    // Update current keyframe
    if (
      state.current_keyframe !== null &&
      state.current_keyframe !== undefined
    ) {
      document.getElementById(
        "currentKF"
      ).textContent = `KF#${state.current_keyframe}`;
    }

    // Update position on map
    if (state.position && mapVisualizer) {
      mapVisualizer.updateRoverPosition(
        state.position.x,
        state.position.y,
        state.position.yaw
      );

      document.getElementById(
        "currentPos"
      ).textContent = `(${state.position.x.toFixed(
        2
      )}, ${state.position.y.toFixed(2)})`;
    }
  }

  startUptimeCounter() {
    setInterval(() => {
      const elapsed = Date.now() - this.startTime;
      const hours = Math.floor(elapsed / 3600000);
      const minutes = Math.floor((elapsed % 3600000) / 60000);
      const seconds = Math.floor((elapsed % 60000) / 1000);

      document.getElementById("uptime").textContent = `${hours
        .toString()
        .padStart(2, "0")}:${minutes.toString().padStart(2, "0")}:${seconds
        .toString()
        .padStart(2, "0")}`;
    }, 1000);
  }

  logActivity(message, type = "info") {
    const log = document.getElementById("activityLog");
    const entry = document.createElement("p");
    entry.className = `log-entry ${type}`;
    entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    log.insertBefore(entry, log.firstChild);

    // Keep only last 50 entries
    while (log.children.length > 50) {
      log.removeChild(log.lastChild);
    }
  }
}

// Create global dashboard instance
let dashboard = null;

document.addEventListener("DOMContentLoaded", () => {
  dashboard = new Dashboard();
});
