import React from 'react';

interface SampleImagesProps {
  onSelectSample: (file: File) => void;
  className?: string;
}

interface SampleImage {
  id: string;
  name: string;
  path: string;
  type: string;
  iconEmoji: string;
}

const SampleImages: React.FC<SampleImagesProps> = ({ onSelectSample, className = '' }) => {
  const sampleImages: SampleImage[] = [
    { 
      id: 'glioma', 
      name: 'Glioma Sample', 
      path: '/sample-images/glioma.jpg', 
      type: 'Glioma',
      iconEmoji: '🔬'
    },
    { 
      id: 'meningioma', 
      name: 'Meningioma Sample', 
      path: '/sample-images/meningioma.jpg', 
      type: 'Meningioma',
      iconEmoji: '🧠'
    },
    { 
      id: 'normal', 
      name: 'Normal Sample', 
      path: '/sample-images/normal.jpg', 
      type: 'Normal',
      iconEmoji: '✅'
    },
    { 
      id: 'pituitary', 
      name: 'Pituitary Sample', 
      path: '/sample-images/pituitary.jpg', 
      type: 'Pituitary',
      iconEmoji: '🔎'
    },
  ];

  const loadSampleImage = async (sample: SampleImage) => {
    try {
      // Fetch the image from the public folder
      const response = await fetch(sample.path);
      const blob = await response.blob();
      
      // Create a File object from the blob
      const file = new File([blob], `${sample.name}.jpg`, { type: 'image/jpeg' });
      
      // Call the parent component's handler
      onSelectSample(file);
    } catch (error) {
      console.error('Error loading sample image:', error);
      alert('Failed to load sample image. Please try again.');
    }
  };

  return (
    <div className={`${className}`}>
      <h3 className="text-md font-medium mb-2 text-gray-700">Sample MRI Images:</h3>
      <div className="grid grid-cols-4 gap-2">
        {sampleImages.map((sample) => (
          <button
            key={sample.id}
            onClick={() => loadSampleImage(sample)}
            className="flex flex-col items-center p-2 border border-gray-300 rounded-md hover:bg-gray-50 transition-colors"
            title={`Load ${sample.name} for testing`}
          >
            <span className="text-2xl mb-1">{sample.iconEmoji}</span>
            <span className="text-xs text-gray-600">{sample.type}</span>
          </button>
        ))}
      </div>
      <p className="text-xs text-gray-500 mt-1">Click on a sample to analyze it</p>
    </div>
  );
};

export default SampleImages;
