import React, { useState } from 'react';

interface XAIVisualizationProps {
  originalImage: string | null;
  explanationImage: string | null;
  overlayImage: string | null;
  xaiMethod: string;
  isLoading: boolean;
  error: string | null;
  onMethodChange: (method: string) => void;
  className?: string;
}

const XAIVisualization: React.FC<XAIVisualizationProps> = ({
  originalImage,
  explanationImage,
  overlayImage,
  xaiMethod,
  isLoading,
  error,
  onMethodChange,
  className = '',
}) => {
  const [overlayOpacity, setOverlayOpacity] = useState(0.7);
  
  const xaiMethods = [
    { id: 'gradcam', name: 'Grad-CAM', description: 'Highlights important regions using class activation mapping' },
    { id: 'scorecam', name: 'Score-CAM', description: 'More precise visualization without requiring gradients' },
    { id: 'shap', name: 'SHAP', description: 'Shows feature importance using Shapley values' },
  ];
  
  return (
    <div className={`rounded-lg border border-gray-200 shadow-sm ${className}`}>
      <div className="p-4">
        <h3 className="text-lg font-medium mb-4 text-gray-800">Model Interpretation</h3>
        
        {/* Method selector */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Explanation Method
          </label>
          <div className="grid grid-cols-3 gap-2">
            {xaiMethods.map((method) => (
              <button
                key={method.id}
                type="button"
                className={`px-4 py-2 text-sm rounded-md transition-colors
                  ${xaiMethod === method.id 
                    ? 'bg-blue-100 text-blue-700 border border-blue-300' 
                    : 'bg-gray-100 text-gray-700 border border-gray-200 hover:bg-gray-200'
                  }`}
                onClick={() => onMethodChange(method.id)}
                disabled={isLoading}
              >
                {method.name}
              </button>
            ))}
          </div>
          <p className="mt-2 text-xs text-gray-500">
            {xaiMethods.find(m => m.id === xaiMethod)?.description}
          </p>
        </div>
        
        {isLoading && (
          <div className="flex items-center justify-center py-8">
            <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-500"></div>
            <p className="ml-3 text-gray-600">Generating visualization...</p>
          </div>
        )}
        
        {error && !isLoading && (
          <div className="bg-red-50 border-l-4 border-red-500 p-4">
            <div className="flex">
              <div className="flex-shrink-0">
                <svg className="h-5 w-5 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
              </div>
              <div className="ml-3">
                <p className="text-sm text-red-700">
                  {error}
                </p>
              </div>
            </div>
          </div>
        )}
        
        {originalImage && overlayImage && !isLoading && !error && (
          <div className="space-y-4">
            <div className="flex flex-col items-center sm:flex-row sm:justify-between sm:space-x-4">
              {/* Original Image */}
              <div className="mb-4 sm:mb-0 sm:w-1/2">
                <p className="text-sm font-medium text-gray-700 mb-2">Original Image</p>
                <div className="rounded border border-gray-200 bg-gray-50 p-1">
                  <img 
                    src={`data:image/png;base64,${originalImage}`} 
                    alt="Original MRI scan" 
                    className="w-full h-auto object-contain rounded max-h-64"
                  />
                </div>
              </div>
              
              {/* Explanation Image */}
              <div className="relative sm:w-1/2">
                <p className="text-sm font-medium text-gray-700 mb-2">
                  {xaiMethod === 'gradcam' ? 'Grad-CAM Visualization' : 
                   xaiMethod === 'scorecam' ? 'Score-CAM Visualization' : 
                   'SHAP Values'}
                </p>
                <div className="rounded border border-gray-200 bg-gray-50 p-1">
                  <img 
                    src={`data:image/png;base64,${overlayImage}`} 
                    alt={`${xaiMethod} explanation overlay`}
                    className="w-full h-auto object-contain rounded max-h-64"
                  />
                </div>
              </div>
            </div>
            
            {/* Opacity Slider */}
            <div>
              <label htmlFor="opacity-slider" className="block text-sm font-medium text-gray-700 mb-2">
                Overlay Opacity: {(overlayOpacity * 100).toFixed(0)}%
              </label>
              <input
                type="range"
                id="opacity-slider"
                min="0"
                max="1"
                step="0.05"
                value={overlayOpacity}
                onChange={(e) => setOverlayOpacity(parseFloat(e.target.value))}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              />
            </div>
            
            {/* XAI Method Info */}
            <div className="mt-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
              <h4 className="text-sm font-medium text-gray-700 mb-2">About this XAI method</h4>
              {xaiMethod === 'gradcam' && (
                <p className="text-sm text-gray-600">
                  Grad-CAM uses the gradients flowing into the final convolutional layer to highlight
                  areas of the image that are most important for the prediction. Red areas indicate
                  regions that strongly influenced the model's decision.
                </p>
              )}
              {xaiMethod === 'scorecam' && (
                <p className="text-sm text-gray-600">
                  Score-CAM produces high-precision visualizations by measuring how different parts of the
                  image affect the output score. It provides more accurate visualizations than Grad-CAM
                  but requires more computation time.
                </p>
              )}
              {xaiMethod === 'shap' && (
                <p className="text-sm text-gray-600">
                  SHAP (SHapley Additive exPlanations) values show how each feature contributes to the
                  model's prediction. This helps identify which regions of the MRI are most indicative
                  of the classification result.
                </p>
              )}
            </div>
          </div>
        )}
        
        {(!originalImage || !overlayImage) && !isLoading && !error && (
          <div className="text-center py-8 text-gray-500">
            <p>Upload an MRI scan and run a prediction to generate a visualization</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default XAIVisualization; 