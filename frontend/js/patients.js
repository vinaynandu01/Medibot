/*
 * Patient Management
 */

class PatientManager {
  constructor() {
    this.patients = [];
    this.currentPatient = null;
    this.modal = document.getElementById("patientModal");
    this.form = document.getElementById("patientForm");

    this.setupEventListeners();
  }

  setupEventListeners() {
    // Add patient button
    document.getElementById("addPatientBtn").addEventListener("click", () => {
      this.showModal();
    });

    // Close modal
    document.getElementById("closeModal").addEventListener("click", () => {
      this.hideModal();
    });

    document.getElementById("cancelBtn").addEventListener("click", () => {
      this.hideModal();
    });

    // Form submit
    this.form.addEventListener("submit", (e) => {
      e.preventDefault();
      this.savePatient();
    });

    // Close modal on outside click
    this.modal.addEventListener("click", (e) => {
      if (e.target === this.modal) {
        this.hideModal();
      }
    });
  }

  async loadPatients() {
    try {
      const response = await api.getPatients();
      if (response.success) {
        this.patients = response.patients;
        this.renderPatients();

        // Update stats
        document.getElementById("patientCount").textContent =
          this.patients.length;

        const totalDeliveries = this.patients.reduce(
          (sum, p) => sum + (p.delivery_count || 0),
          0
        );
        document.getElementById("deliveryCount").textContent = totalDeliveries;
      }
    } catch (error) {
      console.error("Failed to load patients:", error);
      this.showError("Failed to load patients");
    }
  }

  renderPatients() {
    const container = document.getElementById("patientsList");

    if (this.patients.length === 0) {
      container.innerHTML = `
                <div class="patients-empty">
                    <div class="icon">👥</div>
                    <h3>No Patients Yet</h3>
                    <p>Click "Add Patient" to register a new patient</p>
                </div>
            `;
      return;
    }

    container.innerHTML = this.patients
      .map((patient) => this.createPatientCard(patient))
      .join("");

    // Add event listeners
    this.patients.forEach((patient) => {
      // Edit button
      const editBtn = document.querySelector(
        `[data-edit="${patient.patient_id}"]`
      );
      if (editBtn) {
        editBtn.addEventListener("click", () =>
          this.editPatient(patient.patient_id)
        );
      }

      // Delete button
      const deleteBtn = document.querySelector(
        `[data-delete="${patient.patient_id}"]`
      );
      if (deleteBtn) {
        deleteBtn.addEventListener("click", () =>
          this.deletePatient(patient.patient_id)
        );
      }

      // Deliver button
      const deliverBtn = document.querySelector(
        `[data-deliver="${patient.patient_id}"]`
      );
      if (deliverBtn) {
        deliverBtn.addEventListener("click", () =>
          this.deliverToPatient(patient.patient_id)
        );
      }

      // Navigate button
      const navigateBtn = document.querySelector(
        `[data-navigate="${patient.patient_id}"]`
      );
      if (navigateBtn) {
        navigateBtn.addEventListener("click", () =>
          this.navigateToPatient(patient.patient_id)
        );
      }

      // Register face button
      const registerFaceBtn = document.querySelector(
        `[data-register-face="${patient.patient_id}"]`
      );
      if (registerFaceBtn) {
        registerFaceBtn.addEventListener("click", () =>
          this.registerPatientFace(patient.patient_id)
        );
      }

      // Delete face button
      const deleteFaceBtn = document.querySelector(
        `[data-delete-face="${patient.patient_id}"]`
      );
      if (deleteFaceBtn) {
        deleteFaceBtn.addEventListener("click", () =>
          this.deletePatientFace(patient.patient_id)
        );
      }
    });
  }

  createPatientCard(patient) {
    const hasLocation =
      patient.keyframe_id !== null && patient.keyframe_id !== undefined;
    const hasFace = patient.face_registered === true;

    return `
            <div class="patient-card-new fade-in">
                <!-- Patient Avatar and Face Section -->
                <div class="patient-avatar-section">
                    ${
                      hasFace && patient.face_image
                        ? `
                        <div class="patient-face-container">
                            <img src="data:image/jpeg;base64,${patient.face_image}" 
                                 alt="${patient.name}" 
                                 class="patient-face-img" />
                            <div class="face-badge verified">✓ Verified</div>
                        </div>
                    `
                        : `
                        <div class="patient-face-container no-face">
                            <div class="no-face-icon">👤</div>
                            <div class="face-badge pending">⏳ Pending</div>
                        </div>
                    `
                    }
                    
                    <!-- Face Actions -->
                    <div class="face-actions">
                        ${
                          hasFace
                            ? `
                            <button class="face-action-btn delete-face" data-delete-face="${patient.patient_id}" title="Delete Face">
                                <span class="icon">🗑️</span>
                                <span>Remove Face</span>
                            </button>
                        `
                            : `
                            <button class="face-action-btn register-face" data-register-face="${patient.patient_id}" title="Register Face">
                                <span class="icon">📸</span>
                                <span>Register Face</span>
                            </button>
                        `
                        }
                    </div>
                </div>

                <!-- Patient Information Section -->
                <div class="patient-info-section">
                    <div class="patient-main-info">
                        <div class="patient-name-row">
                            <h3 class="patient-name">${this.escapeHtml(
                              patient.name
                            )}</h3>
                            <span class="patient-id">#${
                              patient.patient_id
                            }</span>
                        </div>
                        <div class="patient-room-badge">
                            <span class="icon">🏥</span>
                            Room ${this.escapeHtml(patient.room_number)}
                        </div>
                    </div>

                    <div class="patient-details-grid">
                        <div class="detail-card">
                            <div class="detail-icon">💊</div>
                            <div class="detail-content">
                                <div class="detail-label">Medication</div>
                                <div class="detail-value">${this.escapeHtml(
                                  patient.medication
                                )}</div>
                            </div>
                        </div>

                        <div class="detail-card">
                            <div class="detail-icon">⏰</div>
                            <div class="detail-content">
                                <div class="detail-label">Schedule</div>
                                <div class="detail-value">${this.escapeHtml(
                                  patient.schedule || "Not set"
                                )}</div>
                            </div>
                        </div>

                        ${
                          hasLocation
                            ? `
                        <div class="detail-card">
                            <div class="detail-icon">📍</div>
                            <div class="detail-content">
                                <div class="detail-label">Location</div>
                                <div class="detail-value">Keyframe #${patient.keyframe_id}</div>
                            </div>
                        </div>
                        `
                            : `
                        <div class="detail-card warning">
                            <div class="detail-icon">⚠️</div>
                            <div class="detail-content">
                                <div class="detail-label">Location</div>
                                <div class="detail-value">Not assigned</div>
                            </div>
                        </div>
                        `
                        }

                        <div class="detail-card">
                            <div class="detail-icon">📊</div>
                            <div class="detail-content">
                                <div class="detail-label">Deliveries</div>
                                <div class="detail-value">${
                                  patient.delivery_count || 0
                                } times</div>
                            </div>
                        </div>
                    </div>

                    ${
                      patient.notes
                        ? `
                    <div class="patient-notes-section">
                        <div class="notes-label">📝 Notes:</div>
                        <div class="notes-text">${this.escapeHtml(
                          patient.notes
                        )}</div>
                    </div>
                    `
                        : ""
                    }

                    <!-- Action Buttons Row -->
                    <div class="patient-action-buttons">
                        ${
                          hasLocation
                            ? `
                        <button class="action-btn navigate-btn" data-navigate="${patient.patient_id}">
                            <span class="icon">🧭</span>
                            <span>Navigate</span>
                        </button>
                        `
                            : `
                        <button class="action-btn navigate-btn disabled" disabled title="No location assigned">
                            <span class="icon">🧭</span>
                            <span>Navigate</span>
                        </button>
                        `
                        }
                        <button class="action-btn edit-btn" data-edit="${
                          patient.patient_id
                        }">
                            <span class="icon">✏️</span>
                            <span>Edit</span>
                        </button>
                        <button class="action-btn delete-btn" data-delete="${
                          patient.patient_id
                        }">
                            <span class="icon">🗑️</span>
                            <span>Delete</span>
                        </button>
                    </div>

                    <!-- Last Delivery Info -->
                    ${
                      patient.last_delivery
                        ? `
                    <div class="last-delivery-info">
                        <span class="icon">🕐</span>
                        Last delivery: ${new Date(
                          patient.last_delivery
                        ).toLocaleString("en-US", {
                          month: "short",
                          day: "numeric",
                          hour: "2-digit",
                          minute: "2-digit",
                        })}
                    </div>
                    `
                        : ""
                    }
                </div>
            </div>
        `;
  }

  showModal(patient = null) {
    this.currentPatient = patient;

    // Reset form
    this.form.reset();

    // Populate keyframe dropdown
    this.populateKeyframeDropdown();

    if (patient) {
      // Edit mode
      document.getElementById("modalTitle").textContent = "Edit Patient";
      document.getElementById("patientName").value = patient.name;
      document.getElementById("roomNumber").value = patient.room_number;
      document.getElementById("medication").value = patient.medication;
      document.getElementById("schedule").value = patient.schedule || "";
      document.getElementById("keyframeSelect").value =
        patient.keyframe_id || "";
      document.getElementById("notes").value = patient.notes || "";
    } else {
      // Add mode
      document.getElementById("modalTitle").textContent = "Add New Patient";
    }

    this.modal.classList.add("active");
  }

  hideModal() {
    this.modal.classList.remove("active");
    this.currentPatient = null;
    this.form.reset();
  }

  populateKeyframeDropdown() {
    const select = document.getElementById("keyframeSelect");
    const currentOptions = select.innerHTML;

    // Only populate if keyframes are available
    if (mapVisualizer && mapVisualizer.keyframes.length > 0) {
      select.innerHTML =
        '<option value="">Not assigned</option>' +
        mapVisualizer.keyframes
          .map(
            (kf) =>
              `<option value="${kf.id}">Keyframe #${
                kf.id
              } (${kf.pose[0].toFixed(2)}, ${kf.pose[1].toFixed(2)})</option>`
          )
          .join("");
    }
  }

  async savePatient() {
    const patientData = {
      name: document.getElementById("patientName").value,
      room_number: document.getElementById("roomNumber").value,
      medication: document.getElementById("medication").value,
      schedule: document.getElementById("schedule").value,
      keyframe_id: document.getElementById("keyframeSelect").value
        ? parseInt(document.getElementById("keyframeSelect").value)
        : null,
      notes: document.getElementById("notes").value,
    };

    try {
      let response;
      if (this.currentPatient) {
        // Update existing patient
        response = await api.updatePatient(
          this.currentPatient.patient_id,
          patientData
        );
      } else {
        // Add new patient
        response = await api.addPatient(patientData);
      }

      if (response.success) {
        this.hideModal();
        await this.loadPatients();
        this.showSuccess(
          this.currentPatient
            ? "Patient updated successfully"
            : "Patient added successfully"
        );
      }
    } catch (error) {
      console.error("Failed to save patient:", error);
      this.showError("Failed to save patient: " + error.message);
    }
  }

  async editPatient(patientId) {
    const patient = this.patients.find((p) => p.patient_id === patientId);
    if (patient) {
      this.showModal(patient);
    }
  }

  async deletePatient(patientId) {
    const patient = this.patients.find((p) => p.patient_id === patientId);
    if (!patient) return;

    if (
      !confirm(`Are you sure you want to delete patient "${patient.name}"?`)
    ) {
      return;
    }

    try {
      const response = await api.deletePatient(patientId);
      if (response.success) {
        await this.loadPatients();
        this.showSuccess("Patient deleted successfully");
      }
    } catch (error) {
      console.error("Failed to delete patient:", error);
      this.showError("Failed to delete patient");
    }
  }

  async deliverToPatient(patientId) {
    const patient = this.patients.find((p) => p.patient_id === patientId);
    if (!patient) return;

    if (!patient.keyframe_id && patient.keyframe_id !== 0) {
      this.showError("Patient has no assigned location");
      return;
    }

    if (!confirm(`Start medication delivery to ${patient.name}?`)) {
      return;
    }

    try {
      const response = await api.deliverToPatient(patientId);
      if (response.success) {
        this.showSuccess(`Delivery started to ${patient.name}`);

        // Select keyframe on map
        if (mapVisualizer) {
          mapVisualizer.selectKeyframe(patient.keyframe_id);
        }

        // Log activity
        this.logActivity(
          `Started delivery to ${patient.name} at Keyframe #${patient.keyframe_id}`
        );
      }
    } catch (error) {
      console.error("Failed to start delivery:", error);
      this.showError("Failed to start delivery: " + error.message);
    }
  }

  showSuccess(message) {
    this.showToast(message, "success");
  }

  showError(message) {
    this.showToast(message, "error");
  }

  showToast(message, type = "info") {
    // Simple toast notification
    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    toast.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${
              type === "success"
                ? "#4CAF50"
                : type === "error"
                ? "#F44336"
                : "#2196F3"
            };
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 10000;
            animation: slideInRight 0.3s ease-out;
        `;

    document.body.appendChild(toast);

    setTimeout(() => {
      toast.style.animation = "slideOutRight 0.3s ease-out";
      setTimeout(() => toast.remove(), 300);
    }, CONFIG.TOAST_DURATION);
  }

  logActivity(message) {
    const log = document.getElementById("activityLog");
    const entry = document.createElement("p");
    entry.className = "log-entry";
    entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    log.insertBefore(entry, log.firstChild);

    // Keep only last 50 entries
    while (log.children.length > 50) {
      log.removeChild(log.lastChild);
    }
  }

  registerPatientFace(patientId) {
    // Redirect to face registration page
    window.location.href = `/face_register.html?patient_id=${patientId}`;
  }

  async deletePatientFace(patientId) {
    const patient = this.patients.find((p) => p.patient_id === patientId);
    if (!patient) return;

    if (
      !confirm(
        `Are you sure you want to delete the face registration for ${patient.name}? This cannot be undone.`
      )
    ) {
      return;
    }

    try {
      const response = await api.deleteFace(patientId);
      if (response.success) {
        this.showSuccess(`Face registration deleted for ${patient.name}`);
        await this.loadPatients(); // Reload to update UI
      } else {
        this.showError(
          response.message || "Failed to delete face registration"
        );
      }
    } catch (error) {
      console.error("Failed to delete face:", error);
      this.showError("Failed to delete face registration");
    }
  }

  async navigateToPatient(patientId) {
    const patient = this.patients.find((p) => p.patient_id === patientId);
    if (!patient) return;

    if (!patient.keyframe_id && patient.keyframe_id !== 0) {
      this.showError("Patient has no assigned location");
      return;
    }

    if (!patient.face_registered) {
      if (
        !confirm(
          `${patient.name} does not have face registered. Navigate anyway?`
        )
      ) {
        return;
      }
    }

    if (
      !confirm(
        `Start navigation to ${patient.name} at location #${patient.keyframe_id}?`
      )
    ) {
      return;
    }

    try {
      const response = await api.startNavigation(patientId);
      if (response.success) {
        this.showSuccess(`Navigation started to ${patient.name}`);

        // Select keyframe on map
        if (mapVisualizer) {
          mapVisualizer.selectKeyframe(patient.keyframe_id);
        }

        // Log activity
        this.logActivity(
          `Navigation started to ${patient.name} at Keyframe #${patient.keyframe_id}`
        );
      }
    } catch (error) {
      console.error("Failed to start navigation:", error);
      this.showError("Failed to start navigation: " + error.message);
    }
  }

  escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }
}

// Create global patient manager
let patientManager = null;

document.addEventListener("DOMContentLoaded", () => {
  patientManager = new PatientManager();
});
