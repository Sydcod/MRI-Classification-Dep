import React from 'react';

interface PredictionResultProps {
  prediction: string | null;
  confidence: number | null;
  isLoading: boolean;
  error: string | null;
  className?: string;
}

const PredictionResult: React.FC<PredictionResultProps> = ({
  prediction,
  confidence,
  isLoading,
  error,
  className = '',
}) => {
  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.9) return 'text-green-600';
    if (confidence >= 0.7) return 'text-blue-600';
    if (confidence >= 0.5) return 'text-yellow-600';
    return 'text-red-600';
  };

  // Remove numeric prefix from prediction label
  const displayPrediction = prediction ? prediction.replace(/^\d+/, '') : '';

  return (
    <div className={`rounded-lg border border-gray-200 shadow-sm ${className}`}>
      <div className="p-4">
        <h3 className="text-lg font-medium mb-4 text-gray-800">Analysis Result</h3>
        
        {isLoading && (
          <div className="flex items-center justify-center py-8">
            <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-500"></div>
            <p className="ml-3 text-gray-600">Analyzing image...</p>
          </div>
        )}
        
        {error && !isLoading && (
          <div className="bg-red-50 border-l-4 border-red-500 p-4 mb-4">
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
        
        {prediction && !isLoading && !error && (
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <div>
                <span className="text-gray-500 text-sm">Classification</span>
                <h4 className="text-xl font-semibold">{displayPrediction}</h4>
              </div>
              
              {confidence !== null && (
                <div className="text-right">
                  <span className="text-gray-500 text-sm">Confidence</span>
                  <div className={getConfidenceColor(confidence)}>
                    <span className="text-xl font-semibold">
                      {(confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              )}
            </div>
            
            {/* Confidence bar */}
            {confidence !== null && (
              <div className="mt-2">
                <div className="w-full bg-gray-200 rounded-full h-2.5">
                  <div 
                    className={`h-2.5 rounded-full ${
                      confidence >= 0.9 ? 'bg-green-500' :
                      confidence >= 0.7 ? 'bg-blue-500' :
                      confidence >= 0.5 ? 'bg-yellow-500' :
                      'bg-red-500'
                    }`}
                    style={{ width: `${confidence * 100}%` }}
                  ></div>
                </div>
              </div>
            )}
          </div>
        )}
        
        {!prediction && !isLoading && !error && (
          <div className="text-center py-8 text-gray-500">
            <p>Upload an MRI scan to get a prediction</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default PredictionResult; 