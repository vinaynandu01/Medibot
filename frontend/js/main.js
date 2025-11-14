/*
 * Main Application Entry Point
 * Initializes all modules and coordinates startup
 */

class MedibotApp {
  constructor() {
    this.initialized = false;
    this.sessionToken = null;
    this.user = null;
  }

  async checkAuth() {
    this.sessionToken = localStorage.getItem("session_token");

    if (!this.sessionToken) {
      window.location.href = "/";
      return false;
    }

    try {
      const response = await fetch(
        "http://localhost:5000/api/auth/verify-session",
        {
          headers: {
            Authorization: `Bearer ${this.sessionToken}`,
          },
        }
      );

      const data = await response.json();

      if (!data.success) {
        localStorage.removeItem("session_token");
        localStorage.removeItem("user");
        window.location.href = "/";
        return false;
      }

      this.user = data.user;
      const userNameEl = document.getElementById("userName");
      if (userNameEl) {
        userNameEl.textContent = this.user.full_name;
      }

      return true;
    } catch (error) {
      console.error("Auth check failed:", error);
      window.location.href = "/";
      return false;
    }
  }

  async init() {
    if (this.initialized) return;

    // Check authentication first
    const isAuthenticated = await this.checkAuth();
    if (!isAuthenticated) return;

    console.log("Initializing Medibot application...");

    try {
      // Load map data
      if (mapVisualizer) {
        const mapLoaded = await mapVisualizer.loadMapData();
        if (mapLoaded) {
          console.log("Map data loaded successfully");
        } else {
          console.warn("Failed to load map data");
        }
      }

      // Load patients
      if (patientManager) {
        await patientManager.loadPatients();
        console.log("Patients loaded successfully");
      }

      // Mark as initialized
      this.initialized = true;

      console.log("Medibot application initialized successfully");

      // Log startup
      if (dashboard) {
        dashboard.logActivity("System initialized and ready");
      }
    } catch (error) {
      console.error("Initialization error:", error);
    }
  }

  // Utility functions
  static formatDate(date) {
    return new Date(date).toLocaleString();
  }

  static formatDistance(meters) {
    if (meters < 1) {
      return `${(meters * 100).toFixed(0)} cm`;
    }
    return `${meters.toFixed(2)} m`;
  }

  static formatAngle(radians) {
    const degrees = ((radians * 180) / Math.PI).toFixed(1);
    return `${degrees}°`;
  }
}

// Initialize app when DOM is ready
document.addEventListener("DOMContentLoaded", async () => {
  console.log("DOM loaded, starting Medibot...");

  // Wait for all modules to be ready
  await new Promise((resolve) => setTimeout(resolve, 100));

  // Create and initialize app
  const app = new MedibotApp();
  await app.init();

  // Make app globally available
  window.medibotApp = app;

  // Logout button handler
  const logoutBtn = document.getElementById("logoutBtn");
  if (logoutBtn) {
    logoutBtn.addEventListener("click", async () => {
      const sessionToken = localStorage.getItem("session_token");

      try {
        await fetch("http://localhost:5000/api/auth/logout", {
          method: "POST",
          headers: {
            Authorization: `Bearer ${sessionToken}`,
          },
        });
      } catch (error) {
        console.error("Logout error:", error);
      }

      localStorage.removeItem("session_token");
      localStorage.removeItem("user");
      window.location.href = "/";
    });
  }
});

// Add CSS animations dynamically
const style = document.createElement("style");
style.textContent = `
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(100px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    @keyframes slideOutRight {
        from {
            opacity: 1;
            transform: translateX(0);
        }
        to {
            opacity: 0;
            transform: translateX(100px);
        }
    }
`;
document.head.appendChild(style);

// Error handling
window.addEventListener("error", (e) => {
  console.error("Global error:", e.error);
});

window.addEventListener("unhandledrejection", (e) => {
  console.error("Unhandled promise rejection:", e.reason);
});

// Service worker for offline support (optional)
if ("serviceWorker" in navigator) {
  // Uncomment to enable service worker
  // navigator.serviceWorker.register('/sw.js')
  //     .then(reg => console.log('Service worker registered'))
  //     .catch(err => console.log('Service worker registration failed'));
}
