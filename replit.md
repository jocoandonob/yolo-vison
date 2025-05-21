# YOLO Detection & Segmentation App Guide

## Overview

This repository contains a Streamlit-based web application for object detection and segmentation using various YOLO (You Only Look Once) models. The app allows users to upload images, select different YOLO versions (YOLOv5, YOLOv8, YOLOv9), and perform various computer vision tasks including object detection, semantic segmentation, instance segmentation, and panoptic segmentation.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend

The application uses Streamlit as its frontend framework, which provides an interactive web interface for users to:
- Upload images
- Select YOLO model versions and specific models
- Configure detection parameters
- View detection/segmentation results with visualizations and statistics

Streamlit was chosen for its simplicity in creating data-focused web apps with Python, allowing for rapid development and iteration without requiring frontend expertise.

### Backend

The backend is built in Python and consists of several modules:
- `app.py`: Main application entry point that handles UI rendering and user interactions
- `model_manager.py`: Manages model downloading, loading, and configuration
- `detector.py`: Processes images using the selected YOLO models
- `utils.py`: Provides helper functions for visualization and data formatting

The application is designed to dynamically load the required ML packages (YOLOv5, Ultralytics) only when needed, which helps minimize startup time and memory usage.

## Key Components

### 1. Model Management System

Located in `model_manager.py`, this component:
- Defines mappings between YOLO versions, packages, and model prefixes
- Handles model availability checking and downloading
- Ensures required packages are installed at runtime
- Provides consistent access to models regardless of their underlying implementation

This modular approach allows for easy addition of new YOLO versions or model variants in the future.

### 2. Detection Engine

Located in `detector.py`, this component:
- Processes input images using the selected models
- Adapts to different model versions (YOLOv5, YOLOv8, YOLOv9)
- Supports multiple task types (detection, segmentation)
- Standardizes output format regardless of the underlying model

The detector is designed with a consistent interface while handling the implementation differences between YOLOv5 (which uses a separate package) and YOLOv8/YOLOv9 (which use the Ultralytics package).

### 3. Visualization Tools

Located in `utils.py`, this component:
- Generates visualizations of detection/segmentation results
- Creates statistical analyses of detected objects
- Formats data for presentation in the Streamlit UI
- Provides color coding for different object classes

## Data Flow

1. **User Input**: 
   - User uploads an image through the Streamlit interface
   - User selects YOLO version, model, and task type
   - User adjusts confidence threshold and other parameters

2. **Model Preparation**:
   - System checks if the selected model is available locally
   - If not available, the model is downloaded from the respective repository
   - Required packages are installed if missing

3. **Image Processing**:
   - The uploaded image is passed to the detector
   - The detector loads the appropriate model and processes the image
   - Processing results (bounding boxes, segmentation masks) are returned

4. **Result Visualization**:
   - Detection results are visualized on the original image
   - Statistics and detailed information are displayed
   - Class distribution and confidence metrics are presented as charts

## External Dependencies

### Core Dependencies
- **Streamlit**: Web application framework
- **PyTorch**: Deep learning framework (CPU version for compatibility)
- **YOLOv5**: For YOLOv5 models (`yolov5` package)
- **Ultralytics**: For YOLOv8 and YOLOv9 models
- **NumPy**: For numerical operations
- **Pillow (PIL)**: For image processing
- **Matplotlib**: For visualization
- **Pandas**: For data manipulation and display

The application uses PyTorch CPU version to ensure compatibility across environments, as specified in the package configuration. Models are downloaded at runtime to avoid storing large model files in the repository.

## Deployment Strategy

The application is configured for deployment on Replit with the following settings:

### Runtime Environment
- Python 3.11
- Additional system packages (cairo, ffmpeg, etc.) for image processing
- PyTorch CPU version to ensure compatibility

### Deployment Configuration
- Autoscale deployment target
- Streamlit server runs on port 5000
- Headless server mode with external accessibility

### Execution Workflow
1. The application starts by running `streamlit run app.py --server.port 5000`
2. Streamlit server serves the web interface
3. Models are downloaded on-demand when selected

This deployment strategy balances performance and resource efficiency by:
- Loading models only when needed
- Using CPU-optimized versions of deep learning libraries
- Configuring the Streamlit server for production use

## Development Guidelines

### Adding New Models
1. Update the model mapping in `model_manager.py`
2. Ensure the detector in `detector.py` can handle the new model type
3. Update the UI options in `app.py`

### Performance Considerations
- Models are loaded on-demand to conserve memory
- Consider implementing caching for frequently used models
- For larger images, implement resizing before processing

### Future Enhancements
- Add video processing capabilities
- Implement real-time webcam support
- Add options for model fine-tuning
- Create a gallery of example images