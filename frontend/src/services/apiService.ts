import axios from 'axios';
import config from '../config';

// Base API configuration
const apiClient = axios.create({
  baseURL: config.API_URL,
  headers: {
    'Accept': 'application/json',
  },
  timeout: 120000, // 120 seconds timeout
});

// Types
export interface PredictionResult {
  prediction: string;
  confidence: number;
  probabilities: Record<string, number>;
  model_id: string;
}

export interface ExplanationResult {
  explanation_method: string;
  explanation_name: string;
  explanation_description: string;
  prediction: string;
  confidence: number;
  original_image: string; // Base64 encoded image
  heatmap_image: string; // Base64 encoded image
  overlay_image: string; // Base64 encoded image
  parameters: {
    alpha: number;
    colormap: string;
  }
}

export interface XAIMethod {
  name: string;
  description: string;
}

export interface XAIMethodsResponse {
  methods: Record<string, XAIMethod>;
}

export interface InterpretationResult {
  interpretation: string;
}

// Prediction service
export const predictionService = {
  // Submit an image for prediction
  async predictImage(imageFile: File, modelId: string = 'default'): Promise<PredictionResult> {
    const formData = new FormData();
    formData.append('image', imageFile);
    formData.append('model_id', modelId);
    
    try {
      console.log(`[${config.isProduction ? 'PROD' : 'DEV'}] Sending prediction request to: ${config.API_URL}/predict`);
      console.log('Image file:', imageFile.name, imageFile.type, imageFile.size);
      
      const response = await apiClient.post('/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 180000, // 3 minute timeout for prediction to allow for model loading
      });
      
      console.log(`[${config.isProduction ? 'PROD' : 'DEV'}] Prediction response:`, response.data);
      return response.data;
    } catch (error) {
      console.error('Prediction error:', error);
      if (axios.isAxiosError(error) && error.response) {
        console.error('Response error data:', error.response.data);
        console.error('Response status:', error.response.status);
        throw new Error(error.response.data?.error || `Prediction failed with status ${error.response.status}`);
      }
      throw new Error('Network error occurred during prediction');
    }
  },
  
  // Get explanation for an image
  async explainImage(
    imageFile: File, 
    xaiMethod: string = 'gradcam', 
    modelId: string = 'default',
    alpha: number = 0.5,
    colormap: string = 'jet'
  ): Promise<ExplanationResult> {
    const formData = new FormData();
    formData.append('image', imageFile);
    formData.append('xai_method', xaiMethod);
    formData.append('model_id', modelId);
    formData.append('alpha', alpha.toString());
    formData.append('colormap', colormap);
    
    try {
      const response = await apiClient.post('/explain', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 0, // no timeout for heavy XAI methods
      });
      
      // If the API response doesn't match our expected format, transform it here
      if (!response.data.original_image && response.data.explanation_image) {
        response.data.original_image = response.data.original_image || '';
        response.data.heatmap_image = response.data.explanation_image || '';
        response.data.overlay_image = response.data.overlay_image || response.data.explanation_image || '';
      }
      
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        throw new Error(error.response.data?.error || 'Explanation generation failed');
      }
      throw new Error('Network error occurred during explanation generation');
    }
  },
  
  // Get available XAI methods
  async getXAIMethods(): Promise<XAIMethodsResponse> {
    try {
      const response = await apiClient.get('/methods');
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        throw new Error(error.response.data?.error || 'Failed to get XAI methods');
      }
      throw new Error('Network error occurred while fetching XAI methods');
    }
  },
  
  // Generate AI interpretation using OpenAI Vision
  async generateInterpretation(explanationResult: ExplanationResult): Promise<InterpretationResult> {
    try {
      console.log('Sending interpretation request to:', '/api/interpret');
      
      // Extract the data we need to send to the API
      const interpretationData = {
        original_image: explanationResult.original_image,
        heatmap_image: explanationResult.heatmap_image,
        overlay_image: explanationResult.overlay_image,
        prediction: explanationResult.prediction,
        confidence: explanationResult.confidence
      };
      
      const response = await apiClient.post('/interpret', interpretationData, {
        headers: {
          'Content-Type': 'application/json'
        },
        timeout: 60000 // 60 seconds timeout for longer processing
      });
      
      console.log('Interpretation response:', response.data);
      return response.data;
    } catch (error) {
      console.error('Interpretation error:', error);
      if (axios.isAxiosError(error) && error.response) {
        console.error('Response error data:', error.response.data);
        console.error('Response status:', error.response.status);
        throw new Error(error.response.data?.error || `Interpretation failed with status ${error.response.status}`);
      }
      throw new Error('Network error occurred during interpretation');
    }
  }
};

export default apiClient;
