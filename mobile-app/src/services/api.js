// api.js - Updated with video frame processing
import axios from "axios";
import Constants from "expo-constants";

const getBackendUrl = () => {
  const extra = Constants.expoConfig?.extra || Constants.manifest?.extra || {};
  return extra.backendUrl || "http://localhost:8001";
};

const BACKEND_URL = getBackendUrl();
const API_BASE = `${BACKEND_URL}/api`;

console.log("API connecting to:", BACKEND_URL);

const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
  headers: { "Content-Type": "application/json" },
});

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error("API Request Error:", error);
    return Promise.reject(error);
  }
);

// Response interceptor for logging
api.interceptors.response.use(
  (response) => {
    console.log(
      `API Response: ${response.config.url} - Status: ${response.status}`
    );
    return response;
  },
  (error) => {
    console.error("API Response Error:", error.response?.status, error.message);
    return Promise.reject(error);
  }
);

export const apiService = {
  BACKEND_URL,

  // Get available videos
  getVideos: () => api.get("/videos"),

  // Get video URL for streaming
  getVideoUrl: (videoName) => `${BACKEND_URL}/api/video/${videoName}`,

  // Process video frame at specific timestamp (NEW - Solution 3)
  processVideoFrame: (data) => api.post("/process-video-frame", data),

  // Get all violations
  getViolations: (limit = 500) => api.get("/violations", { params: { limit } }),

  // Log a new violation
  logViolation: (violation) => api.post("/violations", violation),

  // Delete a specific violation
  deleteViolation: (violationId) => api.delete(`/violations/${violationId}`),

  // Reset all alerts
  resetAlerts: () => api.post("/reset-alerts"),

  // Get system statistics
  getStatistics: () => api.get("/statistics"),

  // Health check
  healthCheck: () => api.get("/health"),

  // Status checks
  getStatus: () => api.get("/status"),
  createStatus: (clientName) =>
    api.post("/status", { client_name: clientName }),

  // Legacy frame processing (if you still need it)
  processFrame: (frameData) => api.post("/process-frame", frameData),
};

export default apiService;
