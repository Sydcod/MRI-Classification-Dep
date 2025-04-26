import React, { useState } from 'react';
import ImageUploader from '../components/ImageUploader';
import PredictionResult from '../components/PredictionResult';
import XAIVisualization from '../components/XAIVisualization';
import AIInterpretation from '../components/AIInterpretation';
import { predictionService, PredictionResult as PredictionResultType, ExplanationResult, InterpretationResult } from '../services/apiService';

const PredictPage: React.FC = () => {
  // State for the selected image
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  
  // State for prediction
  const [prediction, setPrediction] = useState<PredictionResultType | null>(null);
  const [isPredicting, setIsPredicting] = useState<boolean>(false);
  const [predictionError, setPredictionError] = useState<string | null>(null);
  
  // State for explanation
  const [explanation, setExplanation] = useState<ExplanationResult | null>(null);
  const [isExplaining, setIsExplaining] = useState<boolean>(false);
  const [explanationError, setExplanationError] = useState<string | null>(null);
  const [xaiMethod, setXaiMethod] = useState<string>('gradcam');
  
  // State for AI interpretation
  const [interpretation, setInterpretation] = useState<string | null>(null);
  const [isInterpreting, setIsInterpreting] = useState<boolean>(false);
  const [interpretationError, setInterpretationError] = useState<string | null>(null);
  
  // Handle image selection
  const handleImageSelect = (file: File) => {
    setSelectedImage(file);
    setPrediction(null);
    setExplanation(null);
    setPredictionError(null);
    setExplanationError(null);
    setInterpretation(null);
    setInterpretationError(null);
  };
  
  // Handle prediction
  const handlePredict = async () => {
    if (!selectedImage) {
      setPredictionError('Please upload an image first');
      return;
    }
    
    setIsPredicting(true);
    setPredictionError(null);
    
    try {
      const result = await predictionService.predictImage(selectedImage);
      setPrediction(result);
      
      // Automatically generate explanation after prediction
      generateExplanation(selectedImage, xaiMethod);
    } catch (error) {
      setPredictionError(error instanceof Error ? error.message : 'Prediction failed');
    } finally {
      setIsPredicting(false);
    }
  };
  
  // Generate explanation
  const generateExplanation = async (image: File, method: string) => {
    setIsExplaining(true);
    setExplanationError(null);
    
    try {
      const result = await predictionService.explainImage(image, method);
      setExplanation(result);
    } catch (error) {
      setExplanationError(error instanceof Error ? error.message : 'Explanation generation failed');
    } finally {
      setIsExplaining(false);
    }
  };
  
  // Handle XAI method change
  const handleMethodChange = (method: string) => {
    setXaiMethod(method);
    
    // Generate new explanation if an image is selected
    if (selectedImage) {
      generateExplanation(selectedImage, method);
    }
  };
  
  // Generate AI interpretation
  const handleGenerateInterpretation = async () => {
    if (!explanation) {
      setInterpretationError('Explanation is required for generating an interpretation');
      return;
    }
    
    setIsInterpreting(true);
    setInterpretationError(null);
    
    try {
      const result = await predictionService.generateInterpretation(explanation);
      setInterpretation(result.interpretation);
    } catch (error) {
      setInterpretationError(error instanceof Error ? error.message : 'Failed to generate interpretation');
    } finally {
      setIsInterpreting(false);
    }
  };
  
  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2 text-gray-800">Brain MRI Classification</h1>
        <p className="text-gray-600">
          Upload a brain MRI scan to classify it and visualize what the model is looking at.
        </p>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Left column - Upload and prediction */}
        <div className="space-y-6">
          <ImageUploader 
            onImageSelect={handleImageSelect}
            accept=".jpg,.jpeg,.png,.tif,.tiff"
            className="mb-4"
          />
          
          <div>
            <button
              onClick={handlePredict}
              disabled={!selectedImage || isPredicting}
              className={`w-full py-3 px-4 rounded-md font-medium transition-colors ${
                !selectedImage || isPredicting
                  ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                  : 'bg-blue-600 text-white hover:bg-blue-700'
              }`}
            >
              {isPredicting ? 'Analyzing...' : 'Analyze Image'}
            </button>
          </div>
          
          <PredictionResult
            prediction={prediction?.prediction || null}
            confidence={prediction?.confidence || null}
            isLoading={isPredicting}
            error={predictionError}
          />
          
          <AIInterpretation
            interpretation={interpretation}
            isLoading={isInterpreting}
            error={interpretationError}
            onRequestInterpretation={handleGenerateInterpretation}
            canRequestInterpretation={!!explanation && !isInterpreting}
            className="mt-6"
          />
        </div>
        
        {/* Right column - Visualization */}
        <div>
          <XAIVisualization
            originalImage={explanation?.original_image || null}
            explanationImage={explanation?.heatmap_image || null}
            overlayImage={explanation?.overlay_image || null}
            xaiMethod={xaiMethod}
            isLoading={isExplaining}
            error={explanationError}
            onMethodChange={handleMethodChange}
          />
        </div>
      </div>
    </div>
  );
};

export default PredictPage;