import axios from 'axios';

// Base API configuration
const apiClient = axios.create({
  baseURL: '/api',
  headers: {
    'Accept': 'application/json',
  },
  timeout: 30000, // 30 seconds timeout
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

// Prediction service
export const predictionService = {
  // Submit an image for prediction
  async predictImage(imageFile: File, modelId: string = 'default'): Promise<PredictionResult> {
    const formData = new FormData();
    formData.append('image', imageFile);
    formData.append('model_id', modelId);
    
    try {
      const response = await apiClient.post('/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response) {
        throw new Error(error.response.data?.error || 'Prediction failed');
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
  }
};

export default apiClient; 