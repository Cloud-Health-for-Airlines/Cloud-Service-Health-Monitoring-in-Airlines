/**
 * BACCP Frontend Configuration
 *
 * Reads backend API base URL from environment variables (VITE_API_URL)
 * with a fallback default. Never hardcodes URLs in components.
 */

export const config = {
  apiUrl: (typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_URL)
    ? import.meta.env.VITE_API_URL.replace(/\/$/, '')
    : 'http://localhost:8000',
  pollIntervalMs: 6000, // 6 seconds polling interval
};
