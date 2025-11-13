/*
 * Main Application Entry Point
 * Initializes all modules and coordinates startup
 */

class MedibotApp {
    constructor() {
        this.initialized = false;
    }

    async init() {
        if (this.initialized) return;

        console.log('Initializing Medibot application...');

        try {
            // Load map data
            if (mapVisualizer) {
                const mapLoaded = await mapVisualizer.loadMapData();
                if (mapLoaded) {
                    console.log('Map data loaded successfully');
                } else {
                    console.warn('Failed to load map data');
                }
            }

            // Load patients
            if (patientManager) {
                await patientManager.loadPatients();
                console.log('Patients loaded successfully');
            }

            // Mark as initialized
            this.initialized = true;

            console.log('Medibot application initialized successfully');

            // Log startup
            if (dashboard) {
                dashboard.logActivity('System initialized and ready');
            }

        } catch (error) {
            console.error('Initialization error:', error);
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
        const degrees = (radians * 180 / Math.PI).toFixed(1);
        return `${degrees}°`;
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', async () => {
    console.log('DOM loaded, starting Medibot...');

    // Wait for all modules to be ready
    await new Promise(resolve => setTimeout(resolve, 100));

    // Create and initialize app
    const app = new MedibotApp();
    await app.init();

    // Make app globally available
    window.medibotApp = app;
});

// Add CSS animations dynamically
const style = document.createElement('style');
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
window.addEventListener('error', (e) => {
    console.error('Global error:', e.error);
});

window.addEventListener('unhandledrejection', (e) => {
    console.error('Unhandled promise rejection:', e.reason);
});

// Service worker for offline support (optional)
if ('serviceWorker' in navigator) {
    // Uncomment to enable service worker
    // navigator.serviceWorker.register('/sw.js')
    //     .then(reg => console.log('Service worker registered'))
    //     .catch(err => console.log('Service worker registration failed'));
}
