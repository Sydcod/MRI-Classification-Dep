import React, { useState, useRef } from 'react';

interface ImageUploaderProps {
  onImageSelect: (file: File) => void;
  accept?: string;
  maxSize?: number;
  className?: string;
  selectedImage?: File | null;
}

const ImageUploader: React.FC<ImageUploaderProps> = ({
  onImageSelect,
  accept = 'image/*',
  maxSize = 5242880, // 5MB
  className = '',
  selectedImage = null,
}) => {
  const [preview, setPreview] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const clearPreviousPreview = () => {
    if (preview) {
      URL.revokeObjectURL(preview);
    }
  };
  
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    handleFiles(files);
  };
  
  const handleFiles = (files: FileList | null) => {
    setError(null);
    clearPreviousPreview();
    
    if (!files || files.length === 0) {
      return;
    }
    
    const file = files[0];
    
    // Check file type
    if (!file.type.match('image.*')) {
      setError('Please upload an image file');
      return;
    }
    
    // Check file size
    if (file.size > maxSize) {
      setError(`File size must be less than ${Math.round(maxSize / 1024 / 1024)}MB`);
      return;
    }
    
    // Create preview
    const previewUrl = URL.createObjectURL(file);
    setPreview(previewUrl);
    onImageSelect(file);
  };
  
  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };
  
  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };
  
  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    const files = e.dataTransfer.files;
    handleFiles(files);
  };
  
  const handleClick = () => {
    fileInputRef.current?.click();
  };
  
  return (
    <div className={className}>
      <div 
        className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors ${
          isDragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-blue-400'
        }`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={handleClick}
      >
        <input 
          type="file" 
          ref={fileInputRef}
          accept={accept}
          onChange={handleFileChange}
          className="hidden"
        />
        
        {selectedImage || preview ? (
          <div className="flex flex-col items-center">
            <img 
              src={selectedImage ? URL.createObjectURL(selectedImage) : preview || ''}
              alt="Preview" 
              className="max-h-64 max-w-full rounded mb-2 object-contain" 
            />
            <p className="text-sm text-gray-500">Click or drop a new image to change</p>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center py-4">
            <svg 
              className="w-12 h-12 text-gray-400 mb-2" 
              fill="none" 
              stroke="currentColor" 
              viewBox="0 0 24 24" 
              xmlns="http://www.w3.org/2000/svg"
            >
              <path 
                strokeLinecap="round" 
                strokeLinejoin="round" 
                strokeWidth={2} 
                d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" 
              />
            </svg>
            <p className="text-gray-700 mb-1 font-medium">Drop your MRI scan here</p>
            <p className="text-sm text-gray-500">or click to browse files</p>
            <p className="text-xs text-gray-400 mt-2">Supported formats: JPG, PNG, TIFF</p>
          </div>
        )}
        
        {error && (
          <div className="mt-2 text-red-500 text-sm">
            {error}
          </div>
        )}
      </div>
    </div>
  );
};

export default ImageUploader; 