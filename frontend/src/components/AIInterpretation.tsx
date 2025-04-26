import React from 'react';

interface AIInterpretationProps {
  interpretation: string | null;
  isLoading: boolean;
  error: string | null;
  onRequestInterpretation: () => void;
  canRequestInterpretation: boolean;
  className?: string;
}

const AIInterpretation: React.FC<AIInterpretationProps> = ({
  interpretation,
  isLoading,
  error,
  onRequestInterpretation,
  canRequestInterpretation,
  className = '',
}) => {
  // Function to format the interpretation text with simple HTML
  const formatInterpretation = (text: string) => {
    // Replace markdown-style headers with styled paragraphs
    let formattedText = text
      .replace(/^# (.*$)/gm, '<h2 class="text-xl font-bold mt-4 mb-2">$1</h2>')
      .replace(/^## (.*$)/gm, '<h3 class="text-lg font-semibold mt-3 mb-1">$1</h3>')
      .replace(/^### (.*$)/gm, '<h4 class="text-base font-medium mt-2 mb-1">$1</h4>')
      // Replace markdown-style lists with HTML lists
      .replace(/^\* (.*$)/gm, '<li>$1</li>')
      .replace(/^- (.*$)/gm, '<li>$1</li>')
      // Replace double line breaks with paragraph breaks
      .replace(/\n\n/g, '</p><p class="my-2">')
      // Replace single line breaks
      .replace(/\n/g, '<br />');
    
    return `<div class="prose prose-sm max-w-none mt-2">${formattedText}</div>`;
  };

  return (
    <div className={`rounded-lg border border-gray-200 shadow-sm ${className}`}>
      <div className="p-4">
        <h3 className="text-lg font-medium mb-4 text-gray-800">AI Medical Interpretation</h3>
        
        {!interpretation && !isLoading && !error && (
          <div className="text-center py-6">
            <div className="mb-4">
              <svg 
                className="w-12 h-12 mx-auto text-gray-400" 
                fill="none" 
                stroke="currentColor" 
                viewBox="0 0 24 24" 
                xmlns="http://www.w3.org/2000/svg"
              >
                <path 
                  strokeLinecap="round" 
                  strokeLinejoin="round" 
                  strokeWidth={2} 
                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" 
                />
              </svg>
            </div>
            <p className="text-gray-600 mb-4">
              Generate an AI-powered medical interpretation of this MRI scan based on the classification and areas of interest.
            </p>
            <button
              onClick={onRequestInterpretation}
              disabled={!canRequestInterpretation}
              className={`px-4 py-2 rounded-md font-medium transition-colors ${
                canRequestInterpretation
                  ? 'bg-indigo-600 text-white hover:bg-indigo-700'
                  : 'bg-gray-300 text-gray-500 cursor-not-allowed'
              }`}
            >
              Generate Interpretation
            </button>
          </div>
        )}
        
        {isLoading && (
          <div className="flex items-center justify-center py-8">
            <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-indigo-500"></div>
            <p className="ml-3 text-gray-600">Generating medical interpretation...</p>
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
        
        {interpretation && !isLoading && !error && (
          <div 
            dangerouslySetInnerHTML={{ __html: formatInterpretation(interpretation) }}
          />
        )}
      </div>
    </div>
  );
};

export default AIInterpretation;
