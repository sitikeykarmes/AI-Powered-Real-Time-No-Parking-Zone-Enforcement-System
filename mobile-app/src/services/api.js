import axios from 'axios';

// Use the same backend URL as the web app
const BACKEND_URL = 'https://mobile-backend-sync.preview.emergentagent.com';
const API_BASE = `${BACKEND_URL}/api`;

const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// API endpoints
export const apiService = {
  // Get available videos
  getVideos: () => api.get('/videos'),
  
  // Get video stream URL
  getVideoUrl: (videoName) => `${BACKEND_URL}/api/video/${videoName}`,
  
  // Get violations/alerts
  getViolations: () => api.get('/violations'),
  
  // Log a new violation
  logViolation: (violation) => api.post('/violations', violation),
  
  // Reset alerts
  resetAlerts: () => api.post('/reset-alerts'),
  
  // Get status checks
  getStatus: () => api.get('/status'),
  
  // Create status check
  createStatus: (clientName) => api.post('/status', { client_name: clientName }),
  
  // Process frame (for mock detection)
  processFrame: (frameData) => api.post('/process-frame', frameData),
};

export default apiService;