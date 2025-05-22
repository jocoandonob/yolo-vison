import streamlit as st
import os
import time
import importlib.util
import model_manager
import utils
from PIL import Image, ImageDraw
import numpy as np
import io
import cv2

def process_image(image, model_version, model_name, task_type, confidence_threshold):
    """Process an image with the selected model."""
    try:
        # Load the model
        model = model_manager.load_model(model_version, model_name)
        if model is None:
            raise Exception(f"Failed to load model: {model_name}")

        # Convert image to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Process based on task type
        if task_type == "Object Detection":
            if model_version == "YOLOv5":
                results = model(img_array)
                # Convert results to a format we can use
                predictions = results.pred[0].cpu().numpy()
                boxes = predictions[:, :4]  # x1, y1, x2, y2
                scores = predictions[:, 4]
                class_ids = predictions[:, 5].astype(int)
                
                # Filter by confidence
                mask = scores >= confidence_threshold
                boxes = boxes[mask]
                scores = scores[mask]
                class_ids = class_ids[mask]
                
                return boxes, scores, class_ids
            else:  # YOLOv8 and YOLOv9
                results = model(img_array, conf=confidence_threshold)[0]
                boxes = results.boxes.xyxy.cpu().numpy()
                scores = results.boxes.conf.cpu().numpy()
                class_ids = results.boxes.cls.cpu().numpy().astype(int)
                return boxes, scores, class_ids
                
        elif task_type == "Semantic Segmentation":
            if model_version == "YOLOv5":
                # YOLOv5 doesn't support segmentation directly
                st.warning("YOLOv5 doesn't support segmentation tasks. Please use YOLOv8 or YOLOv9 for segmentation.")
                return None
            else:  # YOLOv8 and YOLOv9
                results = model(img_array, conf=confidence_threshold)[0]
                if hasattr(results, 'masks') and results.masks is not None:
                    # For semantic segmentation, we combine all masks of the same class
                    masks = results.masks.data.cpu().numpy()  # shape: (N, H, W)
                    class_ids = results.boxes.cls.cpu().numpy().astype(int)
                    
                    # Create a semantic segmentation map
                    semantic_map = np.zeros(img_array.shape[:2], dtype=np.uint8)
                    
                    # Process each mask and combine masks of the same class
                    for mask, class_id in zip(masks, class_ids):
                        # Ensure mask is 2D
                        if len(mask.shape) > 2:
                            mask = np.squeeze(mask)
                        
                        # Resize mask if needed
                        if mask.shape[:2] != img_array.shape[:2]:
                            mask = cv2.resize(mask, (img_array.shape[1], img_array.shape[0]), 
                                            interpolation=cv2.INTER_NEAREST)
                        
                        # Add this mask to the semantic map (class_id + 1 to avoid 0)
                        semantic_map[mask > 0] = class_id + 1
                    
                    return semantic_map, None, None, None
                return None
                
        elif task_type == "Instance Segmentation":
            if model_version == "YOLOv8":
                results = model(img_array, conf=confidence_threshold)[0]
                if hasattr(results, 'masks') and results.masks is not None:
                    # Get segmentation masks
                    masks = results.masks.data.cpu().numpy()  # shape: (N, H, W)
                    boxes = results.boxes.xyxy.cpu().numpy()
                    scores = results.boxes.conf.cpu().numpy()
                    class_ids = results.boxes.cls.cpu().numpy().astype(int)
                    
                    # Process each mask
                    processed_masks = []
                    for mask in masks:
                        # Ensure mask is 2D
                        if len(mask.shape) > 2:
                            mask = np.squeeze(mask)
                        
                        # Resize mask if needed
                        if mask.shape[:2] != img_array.shape[:2]:
                            mask = cv2.resize(mask, (img_array.shape[1], img_array.shape[0]), 
                                            interpolation=cv2.INTER_NEAREST)
                        
                        processed_masks.append(mask)
                    
                    return np.array(processed_masks), boxes, scores, class_ids
            return None
            
    except Exception as e:
        print(f"Error processing image: {e}")
        raise

def display_results(image, results, task_type):
    """Display the processed image with results."""
    if results is None:
        st.warning("No objects detected in the image.")
        return
        
    # Display results based on task type
    if task_type == "Object Detection":
        boxes, scores, class_ids = results
        # Draw bounding boxes
        draw = ImageDraw.Draw(image)
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            # Add label
            label = f"{class_id}: {score:.2f}"
            draw.text((x1, y1-10), label, fill="red")
            
    elif task_type == "Semantic Segmentation":
        semantic_map, _, _, _ = results
        if semantic_map is not None:
            # Convert image to numpy array
            img_array = np.array(image)
            
            # Create RGBA overlay
            overlay = np.zeros((*img_array.shape[:2], 4), dtype=np.uint8)
            overlay[..., :3] = img_array  # Copy RGB channels
            overlay[..., 3] = 255         # Set alpha channel
            
            # Define a set of vibrant colors for different classes
            vibrant_colors = [
                [255, 0, 0, 200],    # Bright Red
                [0, 255, 0, 200],    # Bright Green
                [0, 0, 255, 200],    # Bright Blue
                [255, 255, 0, 200],  # Yellow
                [255, 0, 255, 200],  # Magenta
                [0, 255, 255, 200],  # Cyan
                [255, 128, 0, 200],  # Orange
                [128, 0, 255, 200],  # Purple
                [0, 128, 255, 200],  # Light Blue
                [255, 0, 128, 200],  # Pink
            ]
            
            # Create a colored overlay for each class
            for class_id in range(1, semantic_map.max() + 1):
                mask = semantic_map == class_id
                if mask.any():
                    color = np.array(vibrant_colors[(class_id - 1) % len(vibrant_colors)])
                    overlay[mask] = color
            
            # Convert back to RGB for display
            result_image = overlay[..., :3]
            image = Image.fromarray(result_image)
            
    elif task_type == "Instance Segmentation":
        masks, boxes, scores, class_ids = results
        if masks is not None:
            # Convert image to numpy array
            img_array = np.array(image)
            
            # Create RGBA overlay
            overlay = np.zeros((*img_array.shape[:2], 4), dtype=np.uint8)
            overlay[..., :3] = img_array  # Copy RGB channels
            overlay[..., 3] = 255         # Set alpha channel
            
            # Define a set of vibrant colors for better visibility
            vibrant_colors = [
                [255, 0, 0, 200],    # Bright Red
                [0, 255, 0, 200],    # Bright Green
                [0, 0, 255, 200],    # Bright Blue
                [255, 255, 0, 200],  # Yellow
                [255, 0, 255, 200],  # Magenta
                [0, 255, 255, 200],  # Cyan
                [255, 128, 0, 200],  # Orange
                [128, 0, 255, 200],  # Purple
                [0, 128, 255, 200],  # Light Blue
                [255, 0, 128, 200],  # Pink
            ]
            
            # Create a copy of the overlay for drawing
            draw_overlay = overlay[..., :3].copy()
            
            for i, (mask, box, score, class_id) in enumerate(zip(masks, boxes, scores, class_ids)):
                # Use vibrant colors in sequence
                color = np.array(vibrant_colors[i % len(vibrant_colors)])
                
                # Apply mask directly to overlay
                overlay[mask > 0] = color
                
                # Draw bounding box with thicker lines
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(draw_overlay, (x1, y1), (x2, y2), color[:3].tolist(), 3)
                # Add label with larger font and thicker text
                label = f"{class_id}: {score:.2f}"
                cv2.putText(draw_overlay, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color[:3].tolist(), 2)
            
            # Blend the mask overlay with the drawing overlay
            mask_overlay = overlay[..., :3]
            alpha = 0.7
            result_image = cv2.addWeighted(mask_overlay, alpha, draw_overlay, 1 - alpha, 0)
            image = Image.fromarray(result_image)
    
    # Display the processed image
    st.image(image, caption="Processed Image", use_container_width=True)

# Set page configuration
st.set_page_config(
    page_title="YOLO Detection & Segmentation",
    page_icon="🔍",
    layout="wide"
)

# Check for required packages
torch_available = importlib.util.find_spec("torch") is not None
if not torch_available:
    st.warning("PyTorch (torch) is not installed. Some functionalities may be limited.")

yolov5_available = model_manager.check_package_installed("yolov5") 
ultralytics_available = model_manager.check_package_installed("ultralytics")

# Package status messages
package_status = []
if torch_available:
    package_status.append("✅ PyTorch (torch)")
else:
    package_status.append("❌ PyTorch (torch)")
    
if yolov5_available:
    package_status.append("✅ YOLOv5")
else:
    package_status.append("❌ YOLOv5")
    
if ultralytics_available:
    package_status.append("✅ Ultralytics (YOLOv8/v9)")
else:
    package_status.append("❌ Ultralytics (YOLOv8/v9)")

# Show package status
with st.expander("🔍 Package Status"):
    st.markdown("\n".join(package_status))

if not yolov5_available and not ultralytics_available:
    st.warning("No YOLO packages are installed. Please install 'yolov5' and/or 'ultralytics' packages to use the full functionality.")

# Initialize detector module only if torch is available
detector = None
if torch_available:
    try:
        import detector
    except ImportError:
        st.error("Failed to import detector module. Some dependencies might be missing.")

# Initialize session state for model downloads
if 'downloaded_models' not in st.session_state:
    st.session_state.downloaded_models = model_manager.get_available_models()

# Sidebar for model selection and configuration
with st.sidebar:
    st.header("Model Configuration")
    
    # Get available YOLO versions based on installed packages and models folder
    available_versions = model_manager.get_available_yolo_versions()
    
    if not available_versions:
        st.warning("No YOLO models found. Please install required packages or download models.")
        available_versions = ["YOLOv5", "YOLOv8", "YOLOv9"]  # Show all options but with warnings
    
    # Model type selection with status indicators
    model_version = st.selectbox(
        "Select YOLO Version",
        available_versions,
        help="Select the YOLO model version you want to use"
    )
    
    # Show detailed status for the selected version
    version_info = model_manager.YOLO_VERSIONS[model_version]
    package_name = version_info["package"]
    package_installed = model_manager.check_package_installed(package_name)
    
    # Create status message
    status_message = []
    if package_installed:
        status_message.append(f"✅ Package '{package_name}' is installed")
    else:
        status_message.append(f"❌ Package '{package_name}' is not installed")
    
    # Check for local models
    local_models = [m for m in model_manager.get_available_models() if m.startswith(f"{model_version}/")]
    if local_models:
        status_message.append(f"✅ {len(local_models)} local model(s) found")
        # Show model details in an expander
        with st.expander("View Local Models"):
            for model in local_models:
                model_name = model.split("/")[1]
                model_path = model_manager.get_model_path(model_version, model_name)
                model_info = model_manager.get_model_info(model_path)
                st.markdown(f"""
                **{model_name}**
                - Type: {model_info['type']}
                - Epoch: {model_info['epoch']}
                - Best Fitness: {model_info['best_fitness']}
                """)
    else:
        status_message.append("❌ No local models found")
    
    # Display status
    st.markdown("\n".join(status_message))
    
    # Show warning if the selected model package is not available
    if not package_installed:
        st.warning(f"Package '{package_name}' required for {model_version} is not installed. "
                   f"Models will not work until it's installed.")
    
    # Task type selection
    available_tasks = ["Object Detection"]
    if model_version in ["YOLOv8", "YOLOv9"]:
        available_tasks.extend(["Semantic Segmentation", "Instance Segmentation"])
        
    task_type = st.selectbox(
        "Select Task",
        available_tasks,
        help="Select the type of task to perform"
    )
    
    # Get model names based on version and task
    available_models = model_manager.get_model_names(model_version, task_type)
    
    if not available_models:
        st.warning(f"No models available for {model_version} and {task_type}")
        model_name = None
    else:
        # For YOLOv8 segmentation, ensure we use the correct model name
        if model_version == "YOLOv8" and "Segmentation" in task_type:
            # Filter to only show segmentation models
            available_models = [m for m in available_models if "-seg" in m]
            if not available_models:
                st.warning("No segmentation models available. Please download a segmentation model first.")
                model_name = None
            else:
                model_name = st.selectbox("Select Model", available_models)
        else:
            model_name = st.selectbox("Select Model", available_models)
        
        # Load the selected model
        if model_name:
            try:
                with st.spinner(f"Loading model {model_name}..."):
                    # For YOLOv5 models, show additional information
                    if model_version == "YOLOv5":
                        st.info("Loading YOLOv5 model. This may take a moment...")
                        if not os.path.exists(os.path.join("models", model_name)):
                            st.info("Model not found locally. Attempting to download...")
                    
                    model = model_manager.load_model(model_version, model_name)
                    if model is None:
                        st.error(f"Failed to load model: {model_name}")
                        if model_version == "YOLOv5":
                            st.info("""
                            For YOLOv5 models, you can try:
                            1. Make sure you have a stable internet connection
                            2. Try using a different model size (n, s, m, l, x)
                            3. Check if the model file is corrupted
                            4. Try downloading the model manually from the YOLOv5 repository
                            """)
                    else:
                        st.success(f"Model loaded successfully: {model_name}")
                        # Show model info if available
                        model_path = model_manager.get_model_path(model_version, model_name)
                        model_info = model_manager.get_model_info(model_path)
                        st.info(f"""
                        Model Information:
                        - Type: {model_info['type']}
                        - Epoch: {model_info['epoch']}
                        - Best Fitness: {model_info['best_fitness']}
                        """)
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                if model_version == "YOLOv5":
                    st.info("""
                    If you're having trouble with YOLOv5 models:
                    1. Make sure you have a stable internet connection
                    2. Try using a different model size (n, s, m, l, x)
                    3. Check if the model file is corrupted
                    4. Try downloading the model manually from the YOLOv5 repository
                    """)
    
    # Download section
    st.header("Download Models")
    
    # Check if required packages are available before showing download options
    if (model_version == "YOLOv5" and yolov5_available) or \
       (model_version in ["YOLOv8", "YOLOv9"] and ultralytics_available):
        
        # Show models available for download
        download_options = model_manager.get_downloadable_models(model_version, task_type)
        
        if download_options:
            download_model = st.selectbox("Select Model to Download", download_options)
            
            if st.button("Download Model"):
                if not torch_available:
                    st.error("PyTorch (torch) is required to download models. Please install it first.")
                else:
                    with st.spinner(f"Downloading {download_model}..."):
                        success = model_manager.download_model(model_version, download_model, task_type)
                        if success:
                            st.session_state.downloaded_models = model_manager.get_available_models()
                            st.success(f"Downloaded {download_model} successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to download model. Please ensure the required packages are installed.")
        else:
            st.info("All models for the selected version and task are already downloaded")
    else:
        package_name = model_manager.YOLO_VERSIONS[model_version]["package"]
        st.warning(f"Cannot download models. Package '{package_name}' is not installed.")
    
    # Confidence threshold
    confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

# Main area for image upload and result display
row1 = st.container()

with row1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Input Image")
        # Image upload
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        st.header("Detection Results")
        
        if uploaded_file is not None and model_name:
            # Check if required packages are available
            can_process = torch_available and (yolov5_available or ultralytics_available)
            
            if not can_process:
                if not torch_available:
                    st.error("PyTorch (torch) is required for processing images. Please install it first.")
                else:
                    st.error("Required YOLO packages are not installed. Please install them first.")
            
            # Process button
            elif st.button("Process Image"):
                # Check if the selected package is available
                model_package_available = (model_version == "YOLOv5" and yolov5_available) or \
                                         (model_version in ["YOLOv8", "YOLOv9"] and ultralytics_available)
                
                if not model_package_available:
                    package_name = model_manager.YOLO_VERSIONS[model_version]["package"]
                    st.error(f"Cannot process image. Package '{package_name}' required for {model_version} is not installed.")
                else:
                    try:
                        with st.spinner("Processing..."):
                            # Record start time
                            start_time = time.time()
                            
                            # Process the image
                            results = process_image(
                                image=image,
                                model_version=model_version,
                                model_name=model_name,
                                task_type=task_type,
                                confidence_threshold=confidence
                            )
                            
                            # Calculate processing time
                            processing_time = time.time() - start_time
                            
                            # Display results
                            display_results(image, results, task_type)
                            
                            # Show metrics
                            st.subheader("Results")
                            st.write(f"Processing Time: {processing_time:.3f} seconds")
                            
                            # Display detailed results in expander
                            with st.expander("View Detection Details"):
                                if results is not None:
                                    if task_type == "Object Detection":
                                        boxes, scores, class_ids = results
                                        for box, score, class_id in zip(boxes, scores, class_ids):
                                            st.write(f"Class {class_id}: {score:.2f} confidence")
                                    else:
                                        st.write("Segmentation masks generated")
                                else:
                                    st.write("No objects detected with the current confidence threshold.")
                    
                    except Exception as e:
                        st.error(f"Error processing image: {str(e)}")
                        st.info("""
                        If you're having trouble processing images:
                        1. Make sure the model is loaded correctly
                        2. Try using a different model size
                        3. Adjust the confidence threshold
                        4. Check if the image format is supported
                        """)
        elif uploaded_file is not None and not model_name:
            st.warning("Please select or download a model first")
        else:
            st.info("Please upload an image to begin")
