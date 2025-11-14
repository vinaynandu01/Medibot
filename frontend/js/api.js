/*
 * API Client
 * Handles all HTTP requests to backend
 */

class APIClient {
  constructor(baseURL) {
    this.baseURL = baseURL;
  }

  getAuthHeaders() {
    const token = localStorage.getItem("session_token");
    return {
      "Content-Type": "application/json",
      Authorization: token ? `Bearer ${token}` : "",
    };
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    const config = {
      headers: {
        ...this.getAuthHeaders(),
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.message || "Request failed");
      }

      return data;
    } catch (error) {
      console.error("API Error:", error);
      throw error;
    }
  }

  // Rover API
  async connectRover(ipAddress) {
    return this.request(CONFIG.API_ENDPOINTS.ROVER_CONNECT, {
      method: "POST",
      body: JSON.stringify({ ip_address: ipAddress }),
    });
  }

  async disconnectRover() {
    return this.request(CONFIG.API_ENDPOINTS.ROVER_DISCONNECT, {
      method: "POST",
    });
  }

  async getRoverStatus() {
    return this.request(CONFIG.API_ENDPOINTS.ROVER_STATUS);
  }

  async moveRover(direction) {
    return this.request(CONFIG.API_ENDPOINTS.ROVER_MOVE, {
      method: "POST",
      body: JSON.stringify({ direction }),
    });
  }

  async stopRover() {
    return this.request(CONFIG.API_ENDPOINTS.ROVER_STOP, {
      method: "POST",
    });
  }

  // Map API
  async getKeyframes() {
    return this.request(CONFIG.API_ENDPOINTS.MAP_KEYFRAMES);
  }

  async getMapFeatures() {
    return this.request(CONFIG.API_ENDPOINTS.MAP_FEATURES);
  }

  // Navigation API
  async navigateToKeyframe(keyframeId) {
    return this.request(CONFIG.API_ENDPOINTS.NAV_GOTO, {
      method: "POST",
      body: JSON.stringify({ keyframe_id: keyframeId }),
    });
  }

  // Patient API
  async getPatients() {
    return this.request(CONFIG.API_ENDPOINTS.PATIENTS);
  }

  async getPatient(patientId) {
    return this.request(`${CONFIG.API_ENDPOINTS.PATIENTS}/${patientId}`);
  }

  async addPatient(patientData) {
    return this.request(CONFIG.API_ENDPOINTS.PATIENTS, {
      method: "POST",
      body: JSON.stringify(patientData),
    });
  }

  async updatePatient(patientId, patientData) {
    return this.request(`${CONFIG.API_ENDPOINTS.PATIENTS}/${patientId}`, {
      method: "PUT",
      body: JSON.stringify(patientData),
    });
  }

  async deletePatient(patientId) {
    return this.request(`${CONFIG.API_ENDPOINTS.PATIENTS}/${patientId}`, {
      method: "DELETE",
    });
  }

  async deliverToPatient(patientId) {
    return this.request(CONFIG.API_ENDPOINTS.PATIENT_DELIVER(patientId), {
      method: "POST",
    });
  }

  // Navigation with Face Recognition
  async startNavigation(patientId) {
    return this.request("/api/navigation/start", {
      method: "POST",
      body: JSON.stringify({ patient_id: patientId }),
    });
  }

  async stopNavigation() {
    return this.request("/api/navigation/stop", {
      method: "POST",
    });
  }

  async getNavigationStatus() {
    return this.request("/api/navigation/status");
  }

  // Face Recognition API
  async registerFace(patientId, images) {
    return this.request("/api/face/register", {
      method: "POST",
      body: JSON.stringify({ patient_id: patientId, ...images }),
    });
  }

  async deleteFace(patientId) {
    return this.request(`/api/face/delete/${patientId}`, {
      method: "DELETE",
    });
  }

  async checkFaceRegistration(patientId) {
    return this.request(`/api/face/check-registration/${patientId}`);
  }

  // Camera API
  getCameraFeedURL() {
    return `${this.baseURL}${CONFIG.API_ENDPOINTS.CAMERA_FEED}`;
  }

  getSnapshotURL() {
    return `${this.baseURL}${CONFIG.API_ENDPOINTS.CAMERA_SNAPSHOT}`;
  }
}

// Create global API instance
const api = new APIClient(CONFIG.API_BASE_URL);
