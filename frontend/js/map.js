/*
 * Map Visualization
 * Canvas-based map rendering with keyframes and rover position
 */

class MapVisualizer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');

        this.keyframes = [];
        this.features = [];
        this.roverPosition = null;
        this.targetKeyframe = null;

        // View settings
        this.zoom = 1.0;
        this.offsetX = 0;
        this.offsetY = 0;
        this.bounds = { min_x: 0, max_x: 1, min_y: 0, max_y: 1 };

        // Interaction
        this.isDragging = false;
        this.lastMouseX = 0;
        this.lastMouseY = 0;

        this.setupCanvas();
        this.setupEventListeners();
    }

    setupCanvas() {
        // Set canvas size
        const container = this.canvas.parentElement;
        this.canvas.width = container.clientWidth;
        this.canvas.height = container.clientHeight;

        // High DPI support
        const dpr = window.devicePixelRatio || 1;
        const rect = this.canvas.getBoundingClientRect();

        this.canvas.width = rect.width * dpr;
        this.canvas.height = rect.height * dpr;
        this.ctx.scale(dpr, dpr);

        this.canvas.style.width = rect.width + 'px';
        this.canvas.style.height = rect.height + 'px';
    }

    setupEventListeners() {
        // Mouse events for pan and zoom
        this.canvas.addEventListener('mousedown', this.handleMouseDown.bind(this));
        this.canvas.addEventListener('mousemove', this.handleMouseMove.bind(this));
        this.canvas.addEventListener('mouseup', this.handleMouseUp.bind(this));
        this.canvas.addEventListener('wheel', this.handleWheel.bind(this));

        // Click to select keyframe
        this.canvas.addEventListener('click', this.handleClick.bind(this));

        // Resize
        window.addEventListener('resize', () => {
            this.setupCanvas();
            this.render();
        });
    }

    handleMouseDown(e) {
        this.isDragging = true;
        this.lastMouseX = e.clientX;
        this.lastMouseY = e.clientY;
        this.canvas.style.cursor = 'grabbing';
    }

    handleMouseMove(e) {
        if (this.isDragging) {
            const dx = e.clientX - this.lastMouseX;
            const dy = e.clientY - this.lastMouseY;

            this.offsetX += dx;
            this.offsetY += dy;

            this.lastMouseX = e.clientX;
            this.lastMouseY = e.clientY;

            this.render();
        }
    }

    handleMouseUp(e) {
        this.isDragging = false;
        this.canvas.style.cursor = 'crosshair';
    }

    handleWheel(e) {
        e.preventDefault();

        const delta = e.deltaY > 0 ? 0.9 : 1.1;
        this.zoom *= delta;
        this.zoom = Math.max(0.5, Math.min(5, this.zoom));

        this.render();
    }

    handleClick(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Check if clicked on a keyframe
        for (const kf of this.keyframes) {
            const screenPos = this.worldToScreen(kf.pose[0], kf.pose[1]);
            const dist = Math.sqrt((x - screenPos.x) ** 2 + (y - screenPos.y) ** 2);

            if (dist < CONFIG.MAP.KEYFRAME_SIZE + 5) {
                this.selectKeyframe(kf.id);
                break;
            }
        }
    }

    async loadMapData() {
        try {
            // Load keyframes
            const keyframesResponse = await api.getKeyframes();
            if (keyframesResponse.success) {
                this.keyframes = keyframesResponse.keyframes;
            }

            // Load features
            const featuresResponse = await api.getMapFeatures();
            if (featuresResponse.success) {
                this.features = featuresResponse.features;
            }

            // Calculate bounds
            this.calculateBounds();

            // Initial render
            this.render();

            return true;
        } catch (error) {
            console.error('Failed to load map data:', error);
            return false;
        }
    }

    calculateBounds() {
        if (this.keyframes.length === 0) {
            return;
        }

        const xCoords = this.keyframes.map(kf => kf.pose[0]);
        const yCoords = this.keyframes.map(kf => kf.pose[1]);

        const padding = 0.5;
        this.bounds = {
            min_x: Math.min(...xCoords) - padding,
            max_x: Math.max(...xCoords) + padding,
            min_y: Math.min(...yCoords) - padding,
            max_y: Math.max(...yCoords) + padding
        };
    }

    worldToScreen(worldX, worldY) {
        const canvasWidth = this.canvas.width / (window.devicePixelRatio || 1);
        const canvasHeight = this.canvas.height / (window.devicePixelRatio || 1);

        // Map world coordinates to canvas coordinates
        const mapWidth = this.bounds.max_x - this.bounds.min_x;
        const mapHeight = this.bounds.max_y - this.bounds.min_y;

        // Center the map
        const scale = Math.min(canvasWidth / mapWidth, canvasHeight / mapHeight) * 0.9 * this.zoom;

        const centerX = canvasWidth / 2;
        const centerY = canvasHeight / 2;

        const x = centerX + (worldX - (this.bounds.min_x + mapWidth / 2)) * scale + this.offsetX;
        const y = centerY - (worldY - (this.bounds.min_y + mapHeight / 2)) * scale + this.offsetY;

        return { x, y, scale };
    }

    render() {
        const canvasWidth = this.canvas.width / (window.devicePixelRatio || 1);
        const canvasHeight = this.canvas.height / (window.devicePixelRatio || 1);

        // Clear canvas with white background
        this.ctx.fillStyle = CONFIG.MAP.BACKGROUND_COLOR;
        this.ctx.fillRect(0, 0, canvasWidth, canvasHeight);

        // Draw grid
        this.drawGrid();

        // Draw features (black points)
        this.drawFeatures();

        // Draw keyframes
        this.drawKeyframes();

        // Draw rover position
        if (this.roverPosition) {
            this.drawRover();
        }

        // Draw target
        if (this.targetKeyframe !== null) {
            this.drawTarget();
        }

        // Draw path if navigating
        if (this.roverPosition && this.targetKeyframe !== null) {
            this.drawPath();
        }
    }

    drawGrid() {
        const canvasWidth = this.canvas.width / (window.devicePixelRatio || 1);
        const canvasHeight = this.canvas.height / (window.devicePixelRatio || 1);

        this.ctx.strokeStyle = CONFIG.MAP.GRID_COLOR;
        this.ctx.lineWidth = 1;

        const gridSize = 0.5; // meters
        const startX = Math.floor(this.bounds.min_x / gridSize) * gridSize;
        const endX = Math.ceil(this.bounds.max_x / gridSize) * gridSize;
        const startY = Math.floor(this.bounds.min_y / gridSize) * gridSize;
        const endY = Math.ceil(this.bounds.max_y / gridSize) * gridSize;

        // Vertical lines
        for (let x = startX; x <= endX; x += gridSize) {
            const screen = this.worldToScreen(x, 0);
            this.ctx.beginPath();
            this.ctx.moveTo(screen.x, 0);
            this.ctx.lineTo(screen.x, canvasHeight);
            this.ctx.stroke();
        }

        // Horizontal lines
        for (let y = startY; y <= endY; y += gridSize) {
            const screen = this.worldToScreen(0, y);
            this.ctx.beginPath();
            this.ctx.moveTo(0, screen.y);
            this.ctx.lineTo(canvasWidth, screen.y);
            this.ctx.stroke();
        }
    }

    drawFeatures() {
        this.ctx.fillStyle = CONFIG.MAP.FEATURE_COLOR;

        for (const feature of this.features) {
            const screen = this.worldToScreen(feature.x, feature.y);
            this.ctx.beginPath();
            this.ctx.arc(screen.x, screen.y, CONFIG.MAP.FEATURE_SIZE, 0, Math.PI * 2);
            this.ctx.fill();
        }
    }

    drawKeyframes() {
        for (const kf of this.keyframes) {
            const screen = this.worldToScreen(kf.pose[0], kf.pose[1]);

            // Draw circle
            this.ctx.fillStyle = kf.id === this.targetKeyframe ? CONFIG.MAP.TARGET_COLOR : CONFIG.MAP.KEYFRAME_COLOR;
            this.ctx.strokeStyle = '#FFFFFF';
            this.ctx.lineWidth = 2;

            this.ctx.beginPath();
            this.ctx.arc(screen.x, screen.y, CONFIG.MAP.KEYFRAME_SIZE, 0, Math.PI * 2);
            this.ctx.fill();
            this.ctx.stroke();

            // Draw label
            this.ctx.fillStyle = CONFIG.MAP.TEXT_COLOR;
            this.ctx.font = '10px sans-serif';
            this.ctx.textAlign = 'center';
            this.ctx.textBaseline = 'top';
            this.ctx.fillText(`KF${kf.id}`, screen.x, screen.y + CONFIG.MAP.KEYFRAME_SIZE + 3);
        }
    }

    drawRover() {
        const screen = this.worldToScreen(this.roverPosition.x, this.roverPosition.y);

        // Draw rover as triangle pointing in direction of yaw
        this.ctx.save();
        this.ctx.translate(screen.x, screen.y);
        this.ctx.rotate(-this.roverPosition.yaw); // Negative because canvas Y is inverted

        this.ctx.fillStyle = CONFIG.MAP.ROVER_COLOR;
        this.ctx.strokeStyle = '#FFFFFF';
        this.ctx.lineWidth = 2;

        this.ctx.beginPath();
        this.ctx.moveTo(CONFIG.MAP.ROVER_SIZE, 0);
        this.ctx.lineTo(-CONFIG.MAP.ROVER_SIZE / 2, CONFIG.MAP.ROVER_SIZE / 2);
        this.ctx.lineTo(-CONFIG.MAP.ROVER_SIZE / 2, -CONFIG.MAP.ROVER_SIZE / 2);
        this.ctx.closePath();
        this.ctx.fill();
        this.ctx.stroke();

        this.ctx.restore();
    }

    drawTarget() {
        const kf = this.keyframes.find(k => k.id === this.targetKeyframe);
        if (!kf) return;

        const screen = this.worldToScreen(kf.pose[0], kf.pose[1]);

        // Draw pulsing circle
        this.ctx.strokeStyle = CONFIG.MAP.TARGET_COLOR;
        this.ctx.lineWidth = 3;

        this.ctx.beginPath();
        this.ctx.arc(screen.x, screen.y, CONFIG.MAP.KEYFRAME_SIZE + 5, 0, Math.PI * 2);
        this.ctx.stroke();
    }

    drawPath() {
        // Simple path drawing from rover to target
        const kf = this.keyframes.find(k => k.id === this.targetKeyframe);
        if (!kf) return;

        const start = this.worldToScreen(this.roverPosition.x, this.roverPosition.y);
        const end = this.worldToScreen(kf.pose[0], kf.pose[1]);

        this.ctx.strokeStyle = CONFIG.MAP.ROVER_COLOR;
        this.ctx.lineWidth = 2;
        this.ctx.setLineDash([5, 5]);

        this.ctx.beginPath();
        this.ctx.moveTo(start.x, start.y);
        this.ctx.lineTo(end.x, end.y);
        this.ctx.stroke();

        this.ctx.setLineDash([]);
    }

    updateRoverPosition(x, y, yaw) {
        this.roverPosition = { x, y, yaw };
        this.render();
    }

    selectKeyframe(keyframeId) {
        this.targetKeyframe = keyframeId;
        this.render();

        // Update UI
        document.getElementById('targetKF').textContent = `KF#${keyframeId}`;

        // Dispatch event
        const event = new CustomEvent('keyframe-selected', {
            detail: { keyframeId }
        });
        window.dispatchEvent(event);
    }

    clearSelection() {
        this.targetKeyframe = null;
        this.render();
        document.getElementById('targetKF').textContent = 'None';
    }

    zoomIn() {
        this.zoom *= 1.2;
        this.zoom = Math.min(5, this.zoom);
        this.render();
    }

    zoomOut() {
        this.zoom *= 0.8;
        this.zoom = Math.max(0.5, this.zoom);
        this.render();
    }

    resetView() {
        this.zoom = 1.0;
        this.offsetX = 0;
        this.offsetY = 0;
        this.render();
    }
}

// Create global map instance
let mapVisualizer = null;

// Initialize map when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    mapVisualizer = new MapVisualizer('mapCanvas');
});
