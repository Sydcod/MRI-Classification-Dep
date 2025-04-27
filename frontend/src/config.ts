/**
 * Environment-aware configuration for the MRI Classification application
 * This allows the app to work both in development and production environments
 */

// API configuration
const API_URL = process.env.REACT_APP_API_URL 
  ? (process.env.REACT_APP_API_URL.startsWith('http') 
      ? process.env.REACT_APP_API_URL 
      : `https://${process.env.REACT_APP_API_URL}`)
  : 'http://localhost:5000/api';

// Export configuration
const config = {
  API_URL,
  isProduction: process.env.NODE_ENV === 'production'
};

export default config;
