import axios from 'axios';
import Constants from 'expo-constants';

// Get backend URL from environment variables
// For local development, this will use .env.local
// For production, this will use .env.production
const BACKEND_URL = Constants.expoConfig?.extra?.backendUrl || 
                   process.env.EXPO_PUBLIC_BACKEND_URL || 
                   'http://192.168.1.100:8001'; // Fallback to local

const API_BASE = `${BACKEND_URL}/api`;

console.log('API connecting to:', BACKEND_URL);

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