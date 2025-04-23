# Brain MRI Classification Project - AI Code Generation Guide

## 🤖 Code Generation Strategy

### 1. Project Structure Generation
- Implement modular project structure with clear separation of concerns
- Generate directory scaffolding with recommended layout
- Create template files with basic imports and class structures

#### Example Structure Template
```python
project_structure = {
    "backend": {
        "models": ["densenet169.py", "model_utils.py"],
        "xai": ["gradcam.py", "scorecam.py", "shap_explainer.py"],
        "data": ["preprocessing.py", "augmentation.py", "dataset.py"],
        "utils": ["config.py", "logging_utils.py", "metrics.py"],
        "api": ["__init__.py", "prediction.py", "explanation.py", "model_management.py"]
    },
    "frontend": {
        "components": ["ImageUploader.tsx", "PredictionResult.tsx", "XAIVisualization.tsx"],
        "services": ["predictionService.ts", "xaiService.ts", "modelService.ts"]
    }
}
```

### 2. Model Architecture Generation
- Create dynamic model architecture generation
- Implement transfer learning template
- Generate layer freezing and fine-tuning mechanisms

#### Model Generation Snippet
```python
def generate_densenet_model(num_classes, pretrained=True, grayscale_input=True):
    """
    Dynamically generate DenseNet169 model with custom head
    
    Args:
        num_classes (int): Number of classification classes
        pretrained (bool): Use ImageNet pretrained weights
        grayscale_input (bool): Whether to adapt model for grayscale input
    """
    model = models.densenet169(pretrained=pretrained)
    
    # Handle grayscale input (single channel)
    if grayscale_input:
        # Modify first conv layer to accept single channel
        original_conv = model.features.conv0
        model.features.conv0 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2,
            padding=3, bias=False
        )
        
        if pretrained:
            # If using pretrained weights, adapt the first layer by
            # averaging over the RGB channels
            with torch.no_grad():
                model.features.conv0.weight = nn.Parameter(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )
    
    # Freeze base layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Custom classification head
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    
    return model
```

### 3. Data Processing Pipeline
- Generate comprehensive data preprocessing scripts
- Create dynamic augmentation strategies
- Implement dataset splitting and loading utilities

#### Data Processing Generation
```python
class MRIDatasetGenerator:
    @staticmethod
    def generate_preprocessing_pipeline(mode='train'):
        """
        Generate dynamic image preprocessing transforms for grayscale MRI images
        
        Args:
            mode (str): 'train' to include augmentation, 'val' or 'test' for no augmentation
        """
        if mode == 'train':
            # Apply augmentation only to training data
            return transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                # Correct normalization for grayscale
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            # No augmentation for validation and test sets
            return transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                # Correct normalization for grayscale
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
    
    @staticmethod
    def split_dataset(raw_data_path, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
        """
        Split raw MRI dataset into train/validation/test sets
        
        Args:
            raw_data_path (str): Path to raw MRI images
            output_dir (str): Directory to save split CSV files
            train_ratio, val_ratio, test_ratio (float): Split ratios
            seed (int): Random seed for reproducibility
        """
        # List all image files
        image_files = []
        class_labels = []
        
        # Collect files for each class
        for class_dir in os.listdir(raw_data_path):
            class_path = os.path.join(raw_data_path, class_dir)
            if os.path.isdir(class_path):
                for img_file in os.listdir(class_path):
                    if img_file.endswith(('.jpg', '.png', '.jpeg', '.tif')):
                        rel_path = os.path.join(class_dir, img_file)
                        image_files.append(rel_path)
                        class_labels.append(class_dir)
        
        # Create DataFrame
        df = pd.DataFrame({'file_path': image_files, 'label': class_labels})
        
        # Perform stratified split
        X_train, X_temp, y_train, y_temp = train_test_split(
            df['file_path'], df['label'],
            train_size=train_ratio,
            stratify=df['label'],
            random_state=seed
        )
        
        # Adjust ratios for the validation/test split
        val_test_ratio = val_ratio / (val_ratio + test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            train_size=val_test_ratio,
            stratify=y_temp,
            random_state=seed
        )
        
        # Create and save DataFrames
        train_df = pd.DataFrame({'file_path': X_train, 'label': y_train})
        val_df = pd.DataFrame({'file_path': X_val, 'label': y_val})
        test_df = pd.DataFrame({'file_path': X_test, 'label': y_test})
        
        # Save to CSV
        os.makedirs(output_dir, exist_ok=True)
        train_df.to_csv(os.path.join(output_dir, 'train_dataset.csv'), index=False)
        val_df.to_csv(os.path.join(output_dir, 'val_dataset.csv'), index=False)
        test_df.to_csv(os.path.join(output_dir, 'test_dataset.csv'), index=False)
        
        print(f"Dataset successfully split: "
              f"Training ({len(train_df)} images), "
              f"Validation ({len(val_df)} images), "
              f"Test ({len(test_df)} images)")
        
        return train_df, val_df, test_df
            
    @staticmethod
    def generate_dataset_class():
        """
        Generate a PyTorch Dataset class for brain MRI data
        """
        class BrainMRIDataset(torch.utils.data.Dataset):
            def __init__(self, data_csv, root_dir, transform=None):
                self.data_frame = pd.read_csv(data_csv)
                self.root_dir = root_dir
                self.transform = transform
                
            def __len__(self):
                return len(self.data_frame)
                
            def __getitem__(self, idx):
                img_path = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0])
                image = Image.open(img_path).convert('L')  # Convert to grayscale
                label = self.data_frame.iloc[idx, 1]
                
                if self.transform:
                    image = self.transform(image)
                    
                return image, label
                
        return BrainMRIDataset
```

### 4. Training Loop Generation
- Create adaptable training loop template
- Generate mixed-precision training support
- Implement advanced logging and checkpointing

#### Training Loop Template
```python
def generate_training_loop(model, train_loader, val_loader, config):
    """
    Generate dynamic training loop with advanced features
    
    Features to include:
    - Mixed precision training
    - Learning rate scheduling
    - Early stopping
    - Model checkpointing
    """
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )
    
    # Initialize learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config["scheduler_t0"],
        T_mult=1,
        eta_min=config["min_lr"]
    )
    
    # Initialize focal loss
    criterion = FocalLoss(gamma=config["focal_loss_gamma"])
    
    # Initialize mixed precision training
    scaler = torch.cuda.amp.GradScaler() if config["use_mixed_precision"] else None
    
    # Early stopping setup
    early_stopping = EarlyStopping(
        patience=config["early_stopping_patience"],
        min_delta=config["early_stopping_delta"]
    )
    
    # Training loop implementation
    for epoch in range(config["epochs"]):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(config["device"]), targets.to(config["device"])
            
            # Mixed precision training
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                
                # Gradient clipping
                if config["gradient_clipping"]:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_value"])
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if config["gradient_clipping"]:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_value"])
                
                optimizer.step()
            
            train_loss += loss.item()
            
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(config["device"]), targets.to(config["device"])
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        # Calculate validation metrics
        val_f1 = f1_score(val_targets, val_preds, average='macro')
        
        # Learning rate scheduling
        scheduler.step()
        
        # Early stopping check
        if early_stopping.check(val_loss, model):
            print(f"Early stopping triggered at epoch {epoch}")
            break
            
        # Save checkpoint
        if epoch % config["checkpoint_interval"] == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_f1': val_f1
            }
            torch.save(checkpoint, f"{config['checkpoint_dir']}/checkpoint_epoch{epoch}.pt")
```

### 5. XAI Visualization Generation
- Create multiple XAI explanation generators
- Implement dynamic visualization techniques
- Generate standardized explanation formats

#### XAI Generation Snippet
```python
class XAIExplanationGenerator:
    @staticmethod
    def generate_gradcam_explanation(model, input_tensor, target_layer):
        """
        Dynamically generate Grad-CAM explanation
        
        Args:
            model (nn.Module): Trained model
            input_tensor (torch.Tensor): Input image tensor (grayscale)
            target_layer (nn.Module): Layer to generate CAM from
        """
        # GradCAM implementation for grayscale images
        model.eval()
        
        # Register hooks
        gradients = []
        activations = []
        
        def save_gradient(grad):
            gradients.append(grad)
            
        def save_activation(module, input, output):
            activations.append(output)
            
        # Register hooks
        handle_activation = target_layer.register_forward_hook(save_activation)
        handle_backward = target_layer.register_full_backward_hook(
            lambda module, grad_input, grad_output: save_gradient(grad_output[0])
        )
        
        # Forward pass
        model_output = model(input_tensor)
        pred_class = torch.argmax(model_output).item()
        
        # Backward pass
        model.zero_grad()
        class_score = model_output[0, pred_class]
        class_score.backward()
        
        # Remove hooks
        handle_activation.remove()
        handle_backward.remove()
        
        # Generate GradCAM
        gradients = gradients[0]
        activations = activations[0]
        
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)  # Apply ReLU to focus on positive contributions
        
        # Normalize CAM
        cam = F.interpolate(
            cam, 
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode='bilinear', 
            align_corners=False
        )
        
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
            
        return cam.detach().cpu().numpy()[0, 0]  # Return as numpy array
        
    @staticmethod
    def generate_scorecam_explanation(model, input_tensor, target_layer):
        """
        Dynamically generate Score-CAM explanation
        
        Args:
            model (nn.Module): Trained model
            input_tensor (torch.Tensor): Input image tensor (grayscale)
            target_layer (nn.Module): Layer to generate CAM from
        """
        # ScoreCAM implementation for grayscale images
        # Implementation details here
        pass
        
    @staticmethod
    def generate_shap_explanation(model, input_tensor, background_tensors):
        """
        Generate SHAP values for model explanation
        
        Args:
            model (nn.Module): Trained model
            input_tensor (torch.Tensor): Input image tensor
            background_tensors (torch.Tensor): Background samples for SHAP
        """
        # SHAP implementation
        # Implementation details here
        pass
```

### 6. API Endpoint Generation
- Create template for Flask endpoints
- Generate input validation decorators
- Implement error handling mechanisms

#### API Endpoint Template
```python
def generate_prediction_endpoint(app, model_registry):
    """
    Generate prediction API endpoint with comprehensive error handling
    """
    @app.route('/api/predict', methods=['POST'])
    def predict():
        try:
            # Input validation
            if 'image' not in request.files:
                return jsonify({'error': 'No image file provided'}), 400
                
            image_file = request.files['image']
            if not allowed_file(image_file.filename):
                return jsonify({'error': 'Invalid file format'}), 400
                
            # Get model ID from request
            model_id = request.form.get('model_id', 'default')
            
            # Process the image
            image_tensor = preprocess_image(image_file)
            
            # Get model from registry
            model = model_registry.get_model(model_id)
            if model is None:
                return jsonify({'error': f'Model {model_id} not found'}), 404
                
            # Make prediction
            with torch.no_grad():
                logits = model(image_tensor.to(model.device))
                probabilities = F.softmax(logits, dim=1)
                pred_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][pred_class].item()
                
            # Get class label
            class_labels = model_registry.get_class_labels(model_id)
            prediction = class_labels[pred_class]
            
            # Return result
            return jsonify({
                'prediction': prediction,
                'confidence': confidence,
                'class_id': pred_class,
                'model_id': model_id
            })
            
        except Exception as e:
            app.logger.error(f"Prediction error: {str(e)}")
            return jsonify({'error': 'Internal server error'}), 500


def generate_explanation_endpoint(app, model_registry, xai_factory):
    """
    Generate XAI explanation API endpoint
    """
    @app.route('/api/explain', methods=['POST'])
    def explain():
        try:
            # Input validation
            if 'image' not in request.files:
                return jsonify({'error': 'No image file provided'}), 400
                
            # Get parameters
            image_file = request.files['image']
            model_id = request.form.get('model_id', 'default')
            xai_method = request.form.get('xai_method', 'gradcam')
            
            # Process the image
            image_tensor = preprocess_image(image_file)
            
            # Get model
            model = model_registry.get_model(model_id)
            if model is None:
                return jsonify({'error': f'Model {model_id} not found'}), 404
                
            # Get XAI generator
            xai_generator = xai_factory.get_generator(xai_method)
            if xai_generator is None:
                return jsonify({'error': f'XAI method {xai_method} not supported'}), 400
                
            # Generate explanation
            with torch.no_grad():
                # Get prediction
                logits = model(image_tensor.to(model.device))
                pred_class = torch.argmax(logits, dim=1).item()
                
                # Get target layer
                target_layer = model_registry.get_target_layer(model_id, xai_method)
                
                # Generate explanation
                explanation = xai_generator.generate(
                    model=model,
                    input_tensor=image_tensor.to(model.device),
                    target_layer=target_layer,
                    target_class=pred_class
                )
                
            # Convert explanation to image
            explanation_img = convert_explanation_to_image(explanation, xai_method)
            
            # Encode image to base64
            buffered = BytesIO()
            explanation_img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # Return result
            return jsonify({
                'explanation_image': img_str,
                'prediction': model_registry.get_class_labels(model_id)[pred_class],
                'xai_method': xai_method
            })
            
        except Exception as e:
            app.logger.error(f"Explanation error: {str(e)}")
            return jsonify({'error': 'Internal server error'}), 500


def generate_models_endpoint(app, model_registry):
    """
    Generate model management API endpoints
    """
    @app.route('/api/models', methods=['GET'])
    def list_models():
        """List all available models"""
        try:
            models = model_registry.list_models()
            return jsonify({
                'models': [
                    {
                        'id': model_id,
                        'name': model_info['name'],
                        'description': model_info['description'],
                        'metrics': model_info['metrics']
                    }
                    for model_id, model_info in models.items()
                ]
            })
        except Exception as e:
            app.logger.error(f"List models error: {str(e)}")
            return jsonify({'error': 'Internal server error'}), 500
            
    @app.route('/api/models/<model_id>', methods=['GET'])
    def get_model_details(model_id):
        """Get details for a specific model"""
        try:
            model_info = model_registry.get_model_info(model_id)
            if model_info is None:
                return jsonify({'error': f'Model {model_id} not found'}), 404
                
            return jsonify({
                'id': model_id,
                'name': model_info['name'],
                'description': model_info['description'],
                'metrics': model_info['metrics'],
                'created_at': model_info['created_at'],
                'last_updated': model_info['last_updated'],
                'training_config': model_info['training_config']
            })
        except Exception as e:
            app.logger.error(f"Get model details error: {str(e)}")
            return jsonify({'error': 'Internal server error'}), 500
            
    @app.route('/api/models/upload', methods=['POST'])
    def upload_model():
        """Upload a new model"""
        try:
            # Validate request
            if 'model_file' not in request.files:
                return jsonify({'error': 'No model file provided'}), 400
                
            model_file = request.files['model_file']
            model_info = json.loads(request.form.get('model_info', '{}'))
            
            # Validate model info
            required_fields = ['name', 'description']
            for field in required_fields:
                if field not in model_info:
                    return jsonify({'error': f'Missing required field: {field}'}), 400
                    
            # Save model file
            model_id = model_registry.add_model(model_file, model_info)
            
            return jsonify({
                'id': model_id,
                'message': 'Model uploaded successfully'
            })
        except Exception as e:
            app.logger.error(f"Upload model error: {str(e)}")
            return jsonify({'error': 'Internal server error'}), 500
```

### 7. Frontend Component Generation
- Create React component templates with TypeScript
- Generate service layer for API interactions
- Implement state management patterns

#### Frontend Component Generation
```typescript
function generateImageUploadComponent() {
    /**
     * Generate a React component for image uploading with drag-and-drop functionality
     */
    return `
import React, { useState, useRef } from 'react';
import { useDropzone } from 'react-dropzone';

interface ImageUploaderProps {
  onImageSelect: (file: File) => void;
  accept?: string;
  maxSize?: number;
  className?: string;
}

const ImageUploader: React.FC<ImageUploaderProps> = ({
  onImageSelect,
  accept = 'image/*',
  maxSize = 5242880, // 5MB
  className = '',
}) => {
  const [preview, setPreview] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  
  const clearPreviousPreview = () => {
    if (preview) {
      URL.revokeObjectURL(preview);
    }
  };
  
  const onDrop = (acceptedFiles: File[]) => {
    setError(null);
    clearPreviousPreview();
    
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      const previewUrl = URL.createObjectURL(file);
      setPreview(previewUrl);
      onImageSelect(file);
    }
  };
  
  const onDropRejected = (fileRejections: any[]) => {
    clearPreviousPreview();
    setPreview(null);
    
    if (fileRejections.length > 0) {
      const { errors } = fileRejections[0];
      if (errors.length > 0) {
        setError(errors[0].message);
      }
    }
  };
  
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    onDropRejected,
    accept,
    maxSize,
    multiple: false,
  });
  
  React.useEffect(() => {
    setIsDragging(isDragActive);
  }, [isDragActive]);
  
  return (
    <div className={className}>
      <div 
        {...getRootProps()} 
        className={\`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors \${
          isDragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-blue-400'
        }\`}
      >
        <input {...getInputProps()} />
        
        {preview ? (
          <div className="flex flex-col items-center">
            <img 
              src={preview} 
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
            <p className="text-xs text-gray-400 mt-2">Supported formats: JPG, PNG, DICOM</p>
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
    `;
}

function generatePredictionResultComponent() {
    /**
     * Generate a React component for displaying prediction results
     */
    return `
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
  
  return (
    <div className={\`rounded-lg border border-gray-200 shadow-sm \${className}\`}>
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
                <h4 className="text-xl font-semibold">{prediction}</h4>
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
                    className={\`h-2.5 rounded-full \${
                      confidence >= 0.9 ? 'bg-green-500' :
                      confidence >= 0.7 ? 'bg-blue-500' :
                      confidence >= 0.5 ? 'bg-yellow-500' :
                      'bg-red-500'
                    }\`}
                    style={{ width: \`\${confidence * 100}%\` }}
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
    `;
}

function generateXAIVisualizationComponent() {
    /**
     * Generate a React component for displaying XAI visualizations
     */
    return `
import React, { useState } from 'react';

interface XAIVisualizationProps {
  originalImage: string | null;
  explanationImage: string | null;
  xaiMethod: string;
  isLoading: boolean;
  error: string | null;
  onMethodChange: (method: string) => void;
  className?: string;
}

const XAIVisualization: React.FC<XAIVisualizationProps> = ({
  originalImage,
  explanationImage,
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
    <div className={\`rounded-lg border border-gray-200 shadow-sm \${className}\`}>
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
                className={\`px-4 py-2 text-sm rounded-md transition-colors
                  \${xaiMethod === method.id 
                    ? 'bg-blue-100 text-blue-700 border border-blue-300' 
                    : 'bg-gray-100 text-gray-700 border border-gray-200 hover:bg-gray-200'
                  }\`}
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
        
        {originalImage && explanationImage && !isLoading && !error && (
          <div className="space-y-4">
            <div className="flex flex-col items-center sm:flex-row sm:justify-between sm:space-x-4">
              {/* Original Image */}
              <div className="mb-4 sm:mb-0 sm:w-1/2">
                <p className="text-sm font-medium text-gray-700 mb-2">Original Image</p>
                <div className="rounded border border-gray-200 bg-gray-50 p-1">
                  <img 
                    src={originalImage} 
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
                  <div className="relative">
                    <img 
                      src={originalImage} 
                      alt="Original MRI scan" 
                      className="w-full h-auto object-contain rounded max-h-64"
                    />
                    <img 
                      src={explanationImage} 
                      alt={`\${xaiMethod} explanation`}
                      className="absolute top-0 left-0 w-full h-full object-contain rounded max-h-64"
                      style={{ opacity: overlayOpacity }}
                    />
                  </div>
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
        
        {(!originalImage || !explanationImage) && !isLoading && !error && (
          <div className="text-center py-8 text-gray-500">
            <p>Upload an MRI scan and run a prediction to generate a visualization</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default XAIVisualization;
    `;
}

function generateAPIService() {
    /**
     * Generate TypeScript service for API interactions
     */
    return `
import axios from 'axios';

// Base API configuration
const apiClient = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  },
  timeout: 30000, // 30 seconds timeout
});

// Types
export interface PredictionResult {
  prediction: string;
  confidence: number;
  class_id: number;
  model_id: string;
}

export interface ExplanationResult {
  explanation_image: string; // Base64 encoded image
  prediction: string;
  xai_method: string;
}

export interface ModelInfo {
  id: string;
  name: string;
  description: string;
  metrics: {
    accuracy: number;
    f1_score: number;
  };
  created_at?: string;
  last_updated?: string;
  training_config?: any;
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
      console.error('Prediction error:', error);
      throw error;
    }
  },
  
  // Get explanation for an image
  async explainImage(
    imageFile: File, 
    xaiMethod: string = 'gradcam', 
    modelId: string = 'default'
  ): Promise<ExplanationResult> {
    const formData = new FormData();
    formData.append('image', imageFile);
    formData.append('xai_method', xaiMethod);
    formData.append('model_id', modelId);
    
    try {
      const response = await apiClient.post('/explain', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      return response.data;
    } catch (error) {
      console.error('Explanation error:', error);
      throw error;
    }
  },
};

// Model management service
export const modelService = {
  // Get all available models
  async getModels(): Promise<ModelInfo[]> {
    try {
      const response = await apiClient.get('/models');
      return response.data.models;
    } catch (error) {
      console.error('Get models error:', error);
      throw error;
    }
  },
  
  // Get details for a specific model
  async getModelDetails(modelId: string): Promise<ModelInfo> {
    try {
      const response = await apiClient.get(\`/models/\${modelId}\`);
      return response.data;
    } catch (error) {
      console.error(\`Get model \${modelId} error:\`, error);
      throw error;
    }
  },
  
  // Upload a new model
  async uploadModel(modelFile: File, modelInfo: Partial<ModelInfo>): Promise<{ id: string }> {
    const formData = new FormData();
    formData.append('model_file', modelFile);
    formData.append('model_info', JSON.stringify(modelInfo));
    
    try {
      const response = await apiClient.post('/models/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      return response.data;
    } catch (error) {
      console.error('Upload model error:', error);
      throw error;
    }
  },
};
`;
}
```

## 🧠 AI Generation Considerations
- Maintain consistent code style
- Generate type-safe and well-documented code
- Provide multiple implementation options
- Include comprehensive error handling
- Support easy customization and extension

## 📝 Code Generation Workflow
1. Analyze project requirements
2. Generate modular, extensible code
3. Provide multiple implementation strategies
4. Include comprehensive comments and documentation
5. Generate test case templates

## 🚨 Critical Generation Rules
- Prioritize type safety
- Implement robust error handling
- Ensure code is modular and extensible
- Generate well-documented implementations
- Provide configuration flexibility
- Handle grayscale images correctly throughout the pipeline
- Ensure consistent technology choices (TypeScript for frontend, Python for backend)
- Use Python dictionaries for configuration rather than YAML
- Maintain clear API patterns and endpoint consistency

**Note**: This guide serves as a comprehensive template for AI-powered code generation, focusing on a brain MRI classification project with advanced machine learning and web application components. The examples provided follow best practices for medical imaging applications and incorporate modern software development principles.