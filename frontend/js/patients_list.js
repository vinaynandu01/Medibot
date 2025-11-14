/*
 * Patients Directory List JavaScript
 */

class PatientsDirectory {
  constructor() {
    this.patients = [];
    this.filteredPatients = [];
    this.currentFilter = "all";
    this.modal = document.getElementById("patientModal");
    this.form = document.getElementById("patientForm");
    this.currentPatient = null;

    this.init();
  }

  init() {
    this.checkAuth();
    this.setupEventListeners();
    this.loadPatients();
    this.loadKeyframes();
  }

  async checkAuth() {
    try {
      const response = await api.verifySession();
      if (!response.success) {
        window.location.href = "/";
        return;
      }
      document.getElementById("userName").textContent = response.user.name;
    } catch (error) {
      window.location.href = "/";
    }
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

    // Search
    document.getElementById("searchInput").addEventListener("input", (e) => {
      this.filterPatients(e.target.value);
    });

    // Filter buttons
    document.querySelectorAll(".filter-btn").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        document
          .querySelectorAll(".filter-btn")
          .forEach((b) => b.classList.remove("active"));
        e.target.classList.add("active");
        this.currentFilter = e.target.dataset.filter;
        this.applyFilter();
      });
    });

    // Modal outside click
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
        this.filteredPatients = [...this.patients];
        this.renderPatients();
        this.updateStats();
      }
    } catch (error) {
      console.error("Failed to load patients:", error);
      this.showToast("Failed to load patients", "error");
    }
  }

  async loadKeyframes() {
    try {
      const response = await api.getKeyframes();
      if (response.success) {
        const select = document.getElementById("keyframeId");
        select.innerHTML = '<option value="">No location assigned</option>';
        response.keyframes.forEach((kf) => {
          const option = document.createElement("option");
          option.value = kf.id;
          option.textContent = `Keyframe #${kf.id}`;
          select.appendChild(option);
        });
      }
    } catch (error) {
      console.error("Failed to load keyframes:", error);
    }
  }

  renderPatients() {
    const grid = document.getElementById("patientsGrid");
    const emptyState = document.getElementById("emptyState");

    if (this.filteredPatients.length === 0) {
      grid.style.display = "none";
      emptyState.style.display = "block";
      return;
    }

    grid.style.display = "grid";
    emptyState.style.display = "none";

    grid.innerHTML = this.filteredPatients
      .map((patient) => this.createPatientCard(patient))
      .join("");

    // Add event listeners
    this.filteredPatients.forEach((patient) => {
      // Navigate button
      const navigateBtn = document.querySelector(
        `[data-navigate="${patient.patient_id}"]`
      );
      if (navigateBtn) {
        navigateBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          this.navigateToPatient(patient.patient_id);
        });
      }

      // Edit button
      const editBtn = document.querySelector(
        `[data-edit="${patient.patient_id}"]`
      );
      if (editBtn) {
        editBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          this.editPatient(patient.patient_id);
        });
      }

      // Delete button
      const deleteBtn = document.querySelector(
        `[data-delete="${patient.patient_id}"]`
      );
      if (deleteBtn) {
        deleteBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          this.deletePatient(patient.patient_id);
        });
      }

      // Register face button
      const registerFaceBtn = document.querySelector(
        `[data-register-face="${patient.patient_id}"]`
      );
      if (registerFaceBtn) {
        registerFaceBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          this.registerPatientFace(patient.patient_id);
        });
      }

      // Card click - view details
      const card = document.querySelector(
        `[data-patient-card="${patient.patient_id}"]`
      );
      if (card) {
        card.addEventListener("click", () => {
          this.editPatient(patient.patient_id);
        });
      }
    });
  }

  createPatientCard(patient) {
    const hasLocation =
      patient.keyframe_id !== null && patient.keyframe_id !== undefined;
    const hasFace = patient.face_registered === true;

    return `
            <div class="patient-item" data-patient-card="${patient.patient_id}">
                <div class="patient-image-container">
                    ${
                      hasFace && patient.face_image
                        ? `<img src="data:image/jpeg;base64,${
                            patient.face_image
                          }" alt="${this.escapeHtml(
                            patient.name
                          )}" class="patient-image" />`
                        : `<div class="no-image-placeholder">👤</div>`
                    }
                    <div class="patient-badges">
                        ${
                          hasFace
                            ? `<span class="badge badge-verified">✓ Verified</span>`
                            : `<span class="badge badge-pending">⏳ Pending</span>`
                        }
                        ${
                          hasLocation
                            ? `<span class="badge badge-location">📍 KF#${patient.keyframe_id}</span>`
                            : ""
                        }
                    </div>
                </div>
                <div class="patient-info">
                    <div class="patient-title">
                        <div>
                            <h3 class="patient-name">${this.escapeHtml(
                              patient.name
                            )}</h3>
                            <span class="patient-id">#${
                              patient.patient_id
                            }</span>
                        </div>
                        <span class="patient-room">
                            <span class="icon">🏥</span>
                            ${this.escapeHtml(patient.room_number)}
                        </span>
                    </div>
                    <div class="patient-details">
                        <div class="detail-row">
                            <span class="icon">💊</span>
                            <span class="label">Medication:</span>
                            <span class="value">${this.escapeHtml(
                              patient.medication
                            )}</span>
                        </div>
                        <div class="detail-row">
                            <span class="icon">⏰</span>
                            <span class="label">Schedule:</span>
                            <span class="value">${this.escapeHtml(
                              patient.schedule || "Not set"
                            )}</span>
                        </div>
                        <div class="detail-row">
                            <span class="icon">📊</span>
                            <span class="label">Deliveries:</span>
                            <span class="value">${
                              patient.delivery_count || 0
                            } times</span>
                        </div>
                    </div>
                    <div class="patient-actions">
                        ${
                          !hasFace
                            ? `<button class="action-btn-small register-face" data-register-face="${patient.patient_id}">
                                <span>📸</span> Register Face
                            </button>`
                            : `<button class="action-btn-small navigate" data-navigate="${
                                patient.patient_id
                              }" ${!hasLocation ? "disabled" : ""}>
                                <span>🧭</span> Navigate
                            </button>
                            <button class="action-btn-small edit" data-edit="${
                              patient.patient_id
                            }">
                                <span>✏️</span> Edit
                            </button>
                            <button class="action-btn-small delete" data-delete="${
                              patient.patient_id
                            }">
                                <span>🗑️</span> Delete
                            </button>`
                        }
                    </div>
                </div>
            </div>
        `;
  }

  updateStats() {
    const totalPatients = this.patients.length;
    const verifiedFaces = this.patients.filter((p) => p.face_registered).length;
    const withLocation = this.patients.filter(
      (p) => p.keyframe_id !== null && p.keyframe_id !== undefined
    ).length;
    const totalDeliveries = this.patients.reduce(
      (sum, p) => sum + (p.delivery_count || 0),
      0
    );

    document.getElementById("totalPatients").textContent = totalPatients;
    document.getElementById("verifiedFaces").textContent = verifiedFaces;
    document.getElementById("withLocation").textContent = withLocation;
    document.getElementById("totalDeliveries").textContent = totalDeliveries;
  }

  filterPatients(searchTerm) {
    const term = searchTerm.toLowerCase();
    this.filteredPatients = this.patients.filter(
      (p) =>
        p.name.toLowerCase().includes(term) ||
        p.patient_id.toLowerCase().includes(term) ||
        p.room_number.toLowerCase().includes(term) ||
        p.medication.toLowerCase().includes(term)
    );
    this.applyFilter();
  }

  applyFilter() {
    let filtered = [...this.filteredPatients];

    switch (this.currentFilter) {
      case "verified":
        filtered = filtered.filter((p) => p.face_registered === true);
        break;
      case "pending":
        filtered = filtered.filter((p) => !p.face_registered);
        break;
      case "location":
        filtered = filtered.filter(
          (p) => p.keyframe_id !== null && p.keyframe_id !== undefined
        );
        break;
    }

    this.filteredPatients = filtered;
    this.renderPatients();
  }

  showModal(patient = null) {
    this.currentPatient = patient;
    this.form.reset();

    if (patient) {
      document.getElementById("modalTitle").textContent = "Edit Patient";
      document.getElementById("patientName").value = patient.name;
      document.getElementById("roomNumber").value = patient.room_number;
      document.getElementById("medication").value = patient.medication;
      document.getElementById("schedule").value = patient.schedule || "";
      document.getElementById("keyframeId").value = patient.keyframe_id || "";
      document.getElementById("notes").value = patient.notes || "";
    } else {
      document.getElementById("modalTitle").textContent = "Add New Patient";
    }

    this.modal.classList.add("active");
  }

  hideModal() {
    this.modal.classList.remove("active");
    this.currentPatient = null;
  }

  async savePatient() {
    const patientData = {
      name: document.getElementById("patientName").value,
      room_number: document.getElementById("roomNumber").value,
      medication: document.getElementById("medication").value,
      schedule: document.getElementById("schedule").value,
      keyframe_id: document.getElementById("keyframeId").value || null,
      notes: document.getElementById("notes").value,
    };

    try {
      let response;
      if (this.currentPatient) {
        response = await api.updatePatient(
          this.currentPatient.patient_id,
          patientData
        );
      } else {
        response = await api.addPatient(patientData);
      }

      if (response.success) {
        this.showToast(
          this.currentPatient
            ? "Patient updated successfully"
            : "Patient added successfully",
          "success"
        );
        this.hideModal();
        await this.loadPatients();
      } else {
        this.showToast(response.message || "Failed to save patient", "error");
      }
    } catch (error) {
      console.error("Failed to save patient:", error);
      this.showToast("Failed to save patient", "error");
    }
  }

  editPatient(patientId) {
    const patient = this.patients.find((p) => p.patient_id === patientId);
    if (patient) {
      this.showModal(patient);
    }
  }

  async deletePatient(patientId) {
    const patient = this.patients.find((p) => p.patient_id === patientId);
    if (!patient) return;

    if (
      !confirm(
        `Are you sure you want to delete ${patient.name}? This action cannot be undone.`
      )
    ) {
      return;
    }

    try {
      const response = await api.deletePatient(patientId);
      if (response.success) {
        this.showToast("Patient deleted successfully", "success");
        await this.loadPatients();
      } else {
        this.showToast(response.message || "Failed to delete patient", "error");
      }
    } catch (error) {
      console.error("Failed to delete patient:", error);
      this.showToast("Failed to delete patient", "error");
    }
  }

  registerPatientFace(patientId) {
    window.location.href = `/face_register.html?patient_id=${patientId}`;
  }

  async navigateToPatient(patientId) {
    const patient = this.patients.find((p) => p.patient_id === patientId);
    if (!patient) return;

    if (!patient.keyframe_id && patient.keyframe_id !== 0) {
      this.showToast("Patient has no assigned location", "error");
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
        this.showToast(`Navigation started to ${patient.name}`, "success");
        setTimeout(() => {
          window.location.href = "/dashboard";
        }, 1500);
      } else {
        this.showToast(
          response.message || "Failed to start navigation",
          "error"
        );
      }
    } catch (error) {
      console.error("Failed to start navigation:", error);
      this.showToast("Failed to start navigation", "error");
    }
  }

  showToast(message, type = "info") {
    const toast = document.getElementById("toast");
    toast.textContent = message;
    toast.className = `toast ${type}`;
    toast.classList.add("show");

    setTimeout(() => {
      toast.classList.remove("show");
    }, 3000);
  }

  escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }
}

function logout() {
  localStorage.removeItem("session_token");
  window.location.href = "/";
}

// Initialize
let patientsDirectory;
document.addEventListener("DOMContentLoaded", () => {
  patientsDirectory = new PatientsDirectory();
});
