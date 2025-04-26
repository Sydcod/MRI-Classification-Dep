import React from 'react';
import { Link } from 'react-router-dom';

const DocumentationPage: React.FC = () => {
  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      <div className="mb-8 flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold mb-2 text-gray-800">Documentation</h1>
          <p className="text-gray-600">
            Comprehensive documentation of the Brain MRI Classification system
          </p>
        </div>
        <Link 
          to="/" 
          className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
        >
          Back to App
        </Link>
      </div>

      <div className="prose prose-lg max-w-none">
        <section id="introduction" className="mb-10">
          <h2 className="text-2xl font-bold mb-4 pb-2 border-b border-gray-200">1. Introduction</h2>
          <p>
            This document provides a detailed technical overview of the Brain MRI Classification System, 
            an advanced web application designed to assist medical professionals in analyzing and 
            interpreting brain MRI scans. The system leverages state-of-the-art deep learning 
            techniques to classify brain MRI scans into four categories: glioma, meningioma, 
            pituitary tumors, and normal brain tissue.
          </p>
          <p>
            Beyond classification, the system offers explainability features through Gradient-weighted 
            Class Activation Mapping (Grad-CAM) visualization, which highlights the regions of interest 
            that influenced the model's decision. Additionally, the application integrates the OpenAI 
            Vision API to provide AI-powered medical interpretations of the scan results.
          </p>
        </section>

        <section id="architecture" className="mb-10">
          <h2 className="text-2xl font-bold mb-4 pb-2 border-b border-gray-200">2. System Architecture</h2>
          <p>
            The application follows a client-server architecture with a clear separation of concerns:
          </p>
          
          <h3 className="text-xl font-semibold mt-6 mb-3">2.1 Backend Architecture</h3>
          <p>
            The backend is built using Flask, a lightweight WSGI web application framework in Python.
            It provides RESTful API endpoints that handle image processing, model inference, and
            explainability techniques. The backend is structured as follows:
          </p>
          <ul className="list-disc pl-6 mb-4">
            <li><strong>app.py</strong>: The main application entry point that initializes the Flask app and registers all API blueprints.</li>
            <li><strong>api/</strong>: Directory containing API blueprints:
              <ul className="list-disc pl-6 mt-2">
                <li><strong>prediction.py</strong>: Handles MRI image classification</li>
                <li><strong>explanation.py</strong>: Implements explainability methods like Grad-CAM</li>
                <li><strong>interpretation.py</strong>: Interfaces with OpenAI Vision API for AI medical interpretation</li>
              </ul>
            </li>
            <li><strong>models/</strong>: Contains the DenseNet169 model architecture and pretrained weights.</li>
            <li><strong>utils/</strong>: Utility functions for image preprocessing, model loading, etc.</li>
          </ul>
          
          <h3 className="text-xl font-semibold mt-6 mb-3">2.2 Frontend Architecture</h3>
          <p>
            The frontend is built with React and TypeScript, using modern web development practices:
          </p>
          <ul className="list-disc pl-6 mb-4">
            <li><strong>Components</strong>: Modular, reusable UI components:
              <ul className="list-disc pl-6 mt-2">
                <li><strong>ImageUploader</strong>: Handles image selection and preview</li>
                <li><strong>PredictionResult</strong>: Displays classification results</li>
                <li><strong>XAIVisualization</strong>: Shows explainability visualizations</li>
                <li><strong>AIInterpretation</strong>: Displays the AI-generated medical report</li>
                <li><strong>SampleImages</strong>: Provides sample MRI images for testing</li>
              </ul>
            </li>
            <li><strong>State Management</strong>: React hooks for local component state</li>
            <li><strong>Styling</strong>: Tailwind CSS for responsive and modern UI design</li>
            <li><strong>API Integration</strong>: Axios for RESTful API communication with the backend</li>
          </ul>
        </section>

        <section id="dataset" className="mb-10">
          <h2 className="text-2xl font-bold mb-4 pb-2 border-b border-gray-200">3. Dataset Description</h2>
          <p>
            The model was trained on a dataset of brain MRI scans compiled from multiple sources 
            and carefully curated to ensure quality and diversity. The dataset contains T1-weighted 
            contrast-enhanced MRI scans corresponding to four classes:
          </p>
          <ul className="list-disc pl-6 mb-4">
            <li><strong>Glioma</strong>: A type of tumor that originates in the glial cells of the brain</li>
            <li><strong>Meningioma</strong>: Tumors arising from the meninges, the membranes surrounding the brain and spinal cord</li>
            <li><strong>Pituitary</strong>: Tumors affecting the pituitary gland</li>
            <li><strong>Normal</strong>: Healthy brain tissue without abnormalities</li>
          </ul>
          <p>
            The dataset was preprocessed using the following steps:
          </p>
          <ol className="list-decimal pl-6 mb-4">
            <li><strong>Normalization</strong>: Pixel intensity values were normalized to the range [0, 1]</li>
            <li><strong>Resizing</strong>: Images were resized to 224×224 pixels to match DenseNet169 input requirements</li>
            <li><strong>Data Augmentation</strong>: Random rotations, flips, and slight intensity variations were applied during training to improve model generalization</li>
          </ol>
          <p>
            The dataset was split into training (70%), validation (15%), and test (15%) sets, with stratification to ensure class balance across splits.
          </p>
        </section>

        <section id="model" className="mb-10">
          <h2 className="text-2xl font-bold mb-4 pb-2 border-b border-gray-200">4. Model Architecture and Training</h2>
          
          <h3 className="text-xl font-semibold mt-6 mb-3">4.1 Model Architecture</h3>
          <p>
            The classification model is based on DenseNet169, a convolutional neural network known for its efficiency and performance on image classification tasks. DenseNet features dense connections between layers, which helps mitigate the vanishing gradient problem and strengthens feature propagation.
          </p>
          <p>
            The architecture was modified for our specific task:
          </p>
          <ul className="list-disc pl-6 mb-4">
            <li>Pretrained weights on ImageNet were used as initialization</li>
            <li>The top classification layer was replaced with a new fully connected layer with 4 output neurons (one per class)</li>
            <li>Global average pooling was applied before the final classification layer</li>
            <li>Dropout (rate=0.5) was added before the final layer to reduce overfitting</li>
          </ul>
          
          <h3 className="text-xl font-semibold mt-6 mb-3">4.2 Training Methodology</h3>
          <p>
            The model was trained using the following methodology:
          </p>
          <ul className="list-disc pl-6 mb-4">
            <li><strong>Hardware</strong>: Training was performed on Google Colab using an NVIDIA A100 GPU for accelerated performance</li>
            <li><strong>Optimization</strong>: Adam optimizer with an initial learning rate of 1e-4</li>
            <li><strong>Learning Rate Schedule</strong>: Reduce on plateau with a factor of 0.5 and patience of 5 epochs</li>
            <li><strong>Loss Function</strong>: Focal loss with gamma=2.0 to address class imbalance</li>
            <li><strong>Batch Size</strong>: 16 samples per batch</li>
            <li><strong>Early Stopping</strong>: Training was stopped when validation loss didn't improve for 10 consecutive epochs</li>
            <li><strong>Mixed Precision Training</strong>: Used to accelerate training and reduce memory consumption</li>
          </ul>
          
          <h3 className="text-xl font-semibold mt-6 mb-3">4.3 Performance Metrics</h3>
          <p>
            The model achieved the following performance on the test set:
          </p>
          <table className="border-collapse table-auto w-full text-sm mb-4">
            <thead>
              <tr>
                <th className="border-b font-medium p-4 pl-8 pt-0 pb-3 text-left">Metric</th>
                <th className="border-b font-medium p-4 pl-8 pt-0 pb-3 text-left">Value</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="border-b p-4 pl-8">Accuracy</td>
                <td className="border-b p-4 pl-8">96.90%</td>
              </tr>
              <tr>
                <td className="border-b p-4 pl-8">F1 Score (weighted)</td>
                <td className="border-b p-4 pl-8">0.9686</td>
              </tr>
              <tr>
                <td className="border-b p-4 pl-8">Precision (weighted)</td>
                <td className="border-b p-4 pl-8">0.9692</td>
              </tr>
              <tr>
                <td className="border-b p-4 pl-8">Recall (weighted)</td>
                <td className="border-b p-4 pl-8">0.9690</td>
              </tr>
            </tbody>
          </table>
          <p>
            Per-class metrics indicate robust performance across all categories, with slightly lower 
            performance on the meningioma class, likely due to its visual similarity to certain types of gliomas.
          </p>

          <h3 className="text-xl font-semibold mt-6 mb-3">4.4 Training Convergence</h3>
          <p>
            The training process demonstrated efficient convergence, as shown in the learning curves below. 
            The model was trained on a Google Colab environment with an NVIDIA A100 GPU, which significantly 
            accelerated the training process.
          </p>
          
          <div className="my-8 flex flex-col items-center">
            <img 
              src="/images/training_curves.png" 
              alt="Training and Validation Learning Curves" 
              className="w-full max-w-4xl rounded-lg shadow-md border border-gray-200"
            />
            <p className="text-gray-600 text-sm mt-3 max-w-4xl">
              <strong>Figure 1:</strong> Training and validation learning curves showing model convergence over epochs. 
              The blue lines represent training metrics (accuracy and loss), while the orange lines represent validation metrics. 
              Note that the model reached peak validation accuracy around epoch 13, after which early stopping prevented 
              overfitting. The close alignment between training and validation curves indicates good generalization, with 
              minimal overfitting. The loss curve shows a steady decrease, while accuracy increases rapidly in the first few 
              epochs and then plateaus, which is characteristic of a well-trained deep learning model.
            </p>
          </div>
        </section>

        <section id="xai" className="mb-10">
          <h2 className="text-2xl font-bold mb-4 pb-2 border-b border-gray-200">5. Explainable AI (XAI) Methods</h2>
          <p>
            Explainability is a crucial aspect of medical AI systems. Our application implements 
            Gradient-weighted Class Activation Mapping (Grad-CAM) to provide visual explanations 
            of the model's decision-making process.
          </p>
          
          <h3 className="text-xl font-semibold mt-6 mb-3">5.1 Grad-CAM Implementation</h3>
          <p>
            Grad-CAM works by computing the gradients of the target class with respect to the feature maps 
            of the last convolutional layer. These gradients are global-average-pooled to obtain importance 
            weights for each feature map. The weighted combination of these feature maps is then upsampled 
            to the input image size to create a heatmap highlighting the regions of interest.
          </p>
          <p>
            The implementation in our system follows these steps:
          </p>
          <ol className="list-decimal pl-6 mb-4">
            <li>Forward pass through the model to obtain predictions</li>
            <li>Backpropagate gradients for the predicted class to the target convolutional layer</li>
            <li>Compute importance weights by globally averaging the gradients</li>
            <li>Generate the weighted feature map and apply ReLU to highlight positive influences</li>
            <li>Resize and normalize the resulting heatmap</li>
            <li>Apply a colormap (default: 'jet') for better visualization</li>
            <li>Overlay the heatmap on the original image with adjustable transparency</li>
          </ol>
          
          <h3 className="text-xl font-semibold mt-6 mb-3">5.2 Benefits of Explainability</h3>
          <p>
            The Grad-CAM visualization serves several purposes:
          </p>
          <ul className="list-disc pl-6 mb-4">
            <li>It helps medical professionals understand and validate the model's classification</li>
            <li>It focuses attention on the anatomical regions most relevant for the diagnosis</li>
            <li>It builds trust in the AI system by making its decisions transparent</li>
            <li>It can potentially highlight regions that might be overlooked in manual examination</li>
            <li>It provides educational value for training medical students and residents</li>
          </ul>
        </section>

        <section id="ai-interpretation" className="mb-10">
          <h2 className="text-2xl font-bold mb-4 pb-2 border-b border-gray-200">6. AI Medical Interpretation</h2>
          <p>
            The AI Medical Interpretation feature leverages OpenAI's Vision API (GPT-4o) to generate 
            comprehensive medical reports based on the MRI scan and classification results.
          </p>
          
          <h3 className="text-xl font-semibold mt-6 mb-3">6.1 Implementation Approach</h3>
          <p>
            The interpretation process follows these steps:
          </p>
          <ol className="list-decimal pl-6 mb-4">
            <li>The original MRI image and Grad-CAM overlay are sent to the OpenAI Vision API</li>
            <li>A carefully crafted prompt instructs the model to act as a specialized neuroradiologist</li>
            <li>The prompt includes the AI classification result and confidence score</li>
            <li>The model analyzes the images and generates a structured medical report</li>
            <li>The report is formatted on the frontend for readability</li>
          </ol>
          
          <h3 className="text-xl font-semibold mt-6 mb-3">6.2 Prompt Engineering</h3>
          <p>
            The prompt is designed to elicit a comprehensive medical interpretation with specific sections:
          </p>
          <ul className="list-disc pl-6 mb-4">
            <li>Confirmation or challenge of the AI classification based on visible evidence</li>
            <li>Detailed description of any abnormalities, lesions, or tumors visible in the scan</li>
            <li>Analysis of typical characteristics of the identified condition</li>
            <li>Potential differential diagnoses to consider</li>
            <li>Suggested follow-up tests or imaging that might be needed</li>
            <li>General treatment approaches for the identified condition</li>
          </ul>
          
          <h3 className="text-xl font-semibold mt-6 mb-3">6.3 Clinical Utility</h3>
          <p>
            The AI interpretation feature serves as a decision support tool with several benefits:
          </p>
          <ul className="list-disc pl-6 mb-4">
            <li>It provides a second opinion to complement the radiologist's assessment</li>
            <li>It offers structured analysis that can help in clinical decision-making</li>
            <li>It can help standardize reporting format and comprehensiveness</li>
            <li>It reduces the cognitive load on medical professionals by automating the initial draft of findings</li>
            <li>It can serve as an educational tool for medical trainees</li>
          </ul>
          <p>
            <strong>Important Note:</strong> The AI interpretation is intended to assist medical professionals, 
            not replace their expertise. All findings should be validated by qualified healthcare providers before 
            making clinical decisions.
          </p>
        </section>

        <section id="api" className="mb-10">
          <h2 className="text-2xl font-bold mb-4 pb-2 border-b border-gray-200">7. API Documentation</h2>
          <p>
            The backend exposes the following RESTful API endpoints:
          </p>
          
          <h3 className="text-xl font-semibold mt-6 mb-3">7.1 Prediction API</h3>
          <ul className="list-disc pl-6 mb-4">
            <li><strong>Endpoint</strong>: <code>/api/predict</code></li>
            <li><strong>Method</strong>: POST</li>
            <li><strong>Description</strong>: Classifies a brain MRI scan</li>
            <li><strong>Request</strong>: multipart/form-data with fields:
              <ul className="list-disc pl-6 mt-2">
                <li><code>image</code>: MRI image file</li>
                <li><code>model_id</code> (optional): ID of the model to use (default: "default")</li>
              </ul>
            </li>
            <li><strong>Response</strong>: JSON with classification results:
              <ul className="list-disc pl-6 mt-2">
                <li><code>prediction</code>: Class label (glioma, meningioma, pituitary, or normal)</li>
                <li><code>confidence</code>: Confidence score (0-1)</li>
                <li><code>probabilities</code>: Per-class probability scores</li>
                <li><code>model_id</code>: ID of the model used</li>
              </ul>
            </li>
          </ul>
          
          <h3 className="text-xl font-semibold mt-6 mb-3">7.2 Explanation API</h3>
          <ul className="list-disc pl-6 mb-4">
            <li><strong>Endpoint</strong>: <code>/api/explain</code></li>
            <li><strong>Method</strong>: POST</li>
            <li><strong>Description</strong>: Generates explainability visualizations</li>
            <li><strong>Request</strong>: multipart/form-data with fields:
              <ul className="list-disc pl-6 mt-2">
                <li><code>image</code>: MRI image file</li>
                <li><code>xai_method</code> (optional): Explainability method (default: "gradcam")</li>
                <li><code>model_id</code> (optional): ID of the model to use (default: "default")</li>
                <li><code>alpha</code> (optional): Overlay transparency (0-1, default: 0.5)</li>
                <li><code>colormap</code> (optional): Colormap for heatmap (default: "jet")</li>
              </ul>
            </li>
            <li><strong>Response</strong>: JSON with explanation results:
              <ul className="list-disc pl-6 mt-2">
                <li><code>explanation_method</code>: Method used</li>
                <li><code>explanation_name</code>: Human-readable name of the method</li>
                <li><code>explanation_description</code>: Description of the method</li>
                <li><code>prediction</code>, <code>confidence</code>: Classification results</li>
                <li><code>original_image</code>: Base64-encoded original image</li>
                <li><code>heatmap_image</code>: Base64-encoded heatmap</li>
                <li><code>overlay_image</code>: Base64-encoded overlay of heatmap on original</li>
                <li><code>parameters</code>: Parameters used for visualization</li>
              </ul>
            </li>
          </ul>
          
          <h3 className="text-xl font-semibold mt-6 mb-3">7.3 Interpretation API</h3>
          <ul className="list-disc pl-6 mb-4">
            <li><strong>Endpoint</strong>: <code>/api/interpret</code></li>
            <li><strong>Method</strong>: POST</li>
            <li><strong>Description</strong>: Generates AI medical interpretation</li>
            <li><strong>Request</strong>: JSON with fields:
              <ul className="list-disc pl-6 mt-2">
                <li><code>original_image</code>: Base64-encoded original image</li>
                <li><code>heatmap_image</code>: Base64-encoded heatmap</li>
                <li><code>overlay_image</code>: Base64-encoded overlay</li>
                <li><code>prediction</code>: Class label</li>
                <li><code>confidence</code>: Confidence score</li>
              </ul>
            </li>
            <li><strong>Response</strong>: JSON with interpretation:
              <ul className="list-disc pl-6 mt-2">
                <li><code>interpretation</code>: AI-generated medical report</li>
              </ul>
            </li>
          </ul>
          
          <h3 className="text-xl font-semibold mt-6 mb-3">7.4 Methods API</h3>
          <ul className="list-disc pl-6 mb-4">
            <li><strong>Endpoint</strong>: <code>/api/methods</code></li>
            <li><strong>Method</strong>: GET</li>
            <li><strong>Description</strong>: Retrieves available XAI methods</li>
            <li><strong>Response</strong>: JSON with available methods:
              <ul className="list-disc pl-6 mt-2">
                <li><code>methods</code>: Dictionary of available XAI methods with name and description</li>
              </ul>
            </li>
          </ul>
        </section>

        <section id="future" className="mb-10">
          <h2 className="text-2xl font-bold mb-4 pb-2 border-b border-gray-200">8. Future Improvements</h2>
          <p>
            Several areas of improvement have been identified for future development:
          </p>
          
          <h3 className="text-xl font-semibold mt-6 mb-3">8.1 Model Enhancements</h3>
          <ul className="list-disc pl-6 mb-4">
            <li><strong>Multi-modal Approach</strong>: Incorporate different MRI sequences (T1, T2, FLAIR, etc.) for more robust classification</li>
            <li><strong>3D Convolutional Networks</strong>: Utilize full volumetric data instead of 2D slices</li>
            <li><strong>Ensemble Methods</strong>: Combine multiple models for improved accuracy and robustness</li>
            <li><strong>Expanded Classification</strong>: Add more detailed tumor subtypes and grades</li>
            <li><strong>Segmentation</strong>: Add tumor segmentation capabilities for volumetric analysis</li>
          </ul>
          
          <h3 className="text-xl font-semibold mt-6 mb-3">8.2 Dataset Improvements</h3>
          <ul className="list-disc pl-6 mb-4">
            <li><strong>Larger Dataset</strong>: Acquire more samples for better generalization</li>
            <li><strong>Diverse Demographics</strong>: Include more diverse patient demographics</li>
            <li><strong>Multi-center Data</strong>: Incorporate data from multiple medical centers for robustness to scanner variations</li>
            <li><strong>Annotated Regions</strong>: Include expert-annotated tumor regions for supervised segmentation</li>
          </ul>
          
          <h3 className="text-xl font-semibold mt-6 mb-3">8.3 XAI Advancements</h3>
          <ul className="list-disc pl-6 mb-4">
            <li><strong>Additional XAI Methods</strong>: Implement LIME, SHAP, and other explainability techniques</li>
            <li><strong>Comparative Visualizations</strong>: Show explanations from multiple methods side by side</li>
            <li><strong>Quantitative XAI Metrics</strong>: Add metrics to evaluate explanation quality</li>
            <li><strong>Region-specific Explanations</strong>: Allow users to query specific regions of interest</li>
          </ul>
          
          <h3 className="text-xl font-semibold mt-6 mb-3">8.4 User Experience Improvements</h3>
          <ul className="list-disc pl-6 mb-4">
            <li><strong>Patient Management</strong>: Add patient records and follow-up tracking</li>
            <li><strong>DICOM Integration</strong>: Support direct DICOM file upload and metadata extraction</li>
            <li><strong>Report Generation</strong>: Customizable report templates for clinical use</li>
            <li><strong>Comparison View</strong>: Side-by-side comparison of multiple scans</li>
            <li><strong>Mobile Optimization</strong>: Improved responsive design for mobile devices</li>
          </ul>
          
          <h3 className="text-xl font-semibold mt-6 mb-3">8.5 Deployment Enhancements</h3>
          <ul className="list-disc pl-6 mb-4">
            <li><strong>Model Versioning</strong>: Support for multiple model versions and A/B testing</li>
            <li><strong>Model Monitoring</strong>: Continuous performance monitoring and drift detection</li>
            <li><strong>Federated Learning</strong>: Enable model training across institutions without data sharing</li>
            <li><strong>Containerization</strong>: Dockerize the application for easier deployment</li>
            <li><strong>Authentication & Authorization</strong>: Add user management and role-based access control</li>
          </ul>
        </section>

        <section id="conclusion" className="mb-10">
          <h2 className="text-2xl font-bold mb-4 pb-2 border-b border-gray-200">9. Conclusion</h2>
          <p>
            The Brain MRI Classification System represents a state-of-the-art application of deep learning 
            and explainable AI in neuroradiology. By combining accurate classification, visual explanations, 
            and AI-powered medical interpretation, it offers a comprehensive tool to assist medical 
            professionals in diagnosing brain abnormalities.
          </p>
          <p>
            While the system shows promising performance, it is designed to augment, not replace, the 
            expertise of medical professionals. The future improvements outlined in this document will 
            further enhance its capabilities and clinical utility.
          </p>
          <p>
            As AI continues to advance in the medical field, systems like this exemplify the potential for 
            technology to support healthcare providers in delivering more accurate, efficient, and 
            personalized patient care.
          </p>
        </section>
      </div>
    </div>
  );
};

export default DocumentationPage;
