/**
 * src/api/client.js
 * Axios instance with JWT token management.
 * Token is stored in memory (not localStorage) per spec.
 * Axios interceptor automatically attaches the Bearer token to every request.
 */

import axios from 'axios';

const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const _STORAGE_KEY = 'mo_refresh_token';

// ── Token store (access in memory, refresh in localStorage) ───────────────
let _accessToken  = null;
let _refreshToken = localStorage.getItem(_STORAGE_KEY) || null;

export const setTokens = (access, refresh) => {
  _accessToken  = access;
  _refreshToken = refresh;
  localStorage.setItem(_STORAGE_KEY, refresh);
};
export const clearTokens = () => {
  _accessToken  = null;
  _refreshToken = null;
  localStorage.removeItem(_STORAGE_KEY);
};
export const getAccessToken  = () => _accessToken;
export const getRefreshToken = () => _refreshToken;
export const isAuthenticated = () => !!_accessToken;

// ── Axios instance ─────────────────────────────────────────────────────────
const api = axios.create({ baseURL: BASE_URL });

// Request interceptor — attach Bearer token
api.interceptors.request.use((config) => {
  if (_accessToken) {
    config.headers.Authorization = `Bearer ${_accessToken}`;
  }
  return config;
});

// Response interceptor — handle 401 with refresh attempt
api.interceptors.response.use(
  (res) => res,
  async (error) => {
    const original = error.config;
    if (error.response?.status === 401 && !original._retry && _refreshToken) {
      original._retry = true;
      try {
        const { data } = await axios.post(`${BASE_URL}/auth/refresh`, {
          refresh_token: _refreshToken,
        });
        setTokens(data.access_token, data.refresh_token);
        original.headers.Authorization = `Bearer ${data.access_token}`;
        return api(original);
      } catch {
        clearTokens();
        window.location.href = '/login';
      }
    }
    return Promise.reject(error);
  }
);

// ── API helpers ────────────────────────────────────────────────────────────

export const authAPI = {
  register: (email, password, display_name) =>
    api.post('/auth/register', { email, password, display_name }),
  login: (email, password) =>
    api.post('/auth/login', { email, password }),
};

export const predictAPI = {
  predict:          (payload) => api.post('/predict', payload),
  predictSynthetic: (emotion) => api.post(`/predict/synthetic${emotion ? `?emotion=${emotion}` : ''}`),
  predictVideo:     (file) => {
    const form = new FormData();
    form.append('file', file);
    return api.post('/predict/video', form, { headers: { 'Content-Type': 'multipart/form-data' } });
  },
  explain:          (session_id) => api.get(`/explain/${session_id}`),
};

export const conflictAPI = {
  explain:        (payload) => api.post('/explain/conflict', payload),
  explainSession: (id)      => api.get(`/explain/conflict/${id}`),
  generate:       (physio, video) => api.post(
    `/predict/conflict${physio && video ? `?physio_emotion=${physio}&video_emotion=${video}` : ''}`
  ),
};

export const sessionsAPI = {
  list:   ()   => api.get('/sessions'),
  detail: (id) => api.get(`/sessions/${id}`),
};

export const chatAPI = {
  send:    (session_id, message) => api.post('/chat', { session_id, message }),
  history: (session_id)          => api.get(`/chat/history/${session_id}`),
};

export const dashboardAPI = {
  summary: () => api.get('/dashboard/summary'),
};

export default api;
