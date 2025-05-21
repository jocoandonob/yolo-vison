import io
import cv2
import numpy as np
import torch
from PIL import Image
import model_manager
import utils

def process_image(img_bytes, model_version, model_name, task_type, confidence_threshold=0.5):
    """
    Process an image using the selected YOLO model for the specified task.
    
    Args:
        img_bytes: Image bytes to process
        model_version: YOLO version (YOLOv5, YOLOv8, YOLOv9)
        model_name: Name of the model to use
        task_type: Type of task (Object Detection, Semantic Segmentation, etc.)
        confidence_threshold: Confidence threshold for detections
        
    Returns:
        Tuple of (processed_image, detection_results)
    """
    # Ensure required packages are installed
    package_name = model_manager.YOLO_VERSIONS[model_version]["package"]
    model_manager.ensure_package_installed(package_name)
    
    # Convert bytes to image
    image = Image.open(io.BytesIO(img_bytes))
    image_np = np.array(image)
    
    # Get full model path
    model_path = model_manager.get_model_path(model_version, model_name)
    
    # Process based on model version
    if model_version == "YOLOv5":
        return process_with_yolov5(image_np, model_path, task_type, confidence_threshold)
    else:  # YOLOv8 and YOLOv9
        return process_with_ultralytics(image_np, model_path, task_type, confidence_threshold)

def process_with_yolov5(image_np, model_path, task_type, confidence_threshold):
    """Process image with YOLOv5."""
    import yolov5
    
    # Load model
    model = yolov5.load(model_path)
    model.conf = confidence_threshold  # Set confidence threshold
    
    # Run inference
    results = model(image_np)
    
    # Process results based on task type
    if task_type == "Object Detection":
        # Get detection results
        detections = []
        for pred in results.pred[0]:
            x1, y1, x2, y2, conf, cls = pred.tolist()
            class_id = int(cls)
            class_name = results.names[class_id]
            detections.append({
                'class_id': class_id,
                'class_name': class_name,
                'confidence': float(conf),
                'bbox': [float(x1), float(y1), float(x2), float(y2)]
            })
        
        # Render image with results
        rendered_img = results.render()[0]
        
        return Image.fromarray(rendered_img), detections
    
    elif "Segmentation" in task_type:
        # For segmentation tasks in YOLOv5
        try:
            # Try to access masks if available
            masks = results.pred[0].masks.data
            
            # Get segmentation results
            segmentations = []
            for i, mask in enumerate(masks):
                pred = results.pred[0][i]
                x1, y1, x2, y2, conf, cls = pred.tolist()
                class_id = int(cls)
                class_name = results.names[class_id]
                
                segmentations.append({
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': float(conf),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'mask': mask.cpu().numpy()
                })
            
            # Render image with results
            rendered_img = results.render()[0]
            
            return Image.fromarray(rendered_img), segmentations
        except:
            # If no masks are available, fall back to detection
            return process_with_yolov5(image_np, model_path, "Object Detection", confidence_threshold)
    
    # Default fallback
    rendered_img = results.render()[0]
    return Image.fromarray(rendered_img), []

def process_with_ultralytics(image_np, model_path, task_type, confidence_threshold):
    """Process image with YOLOv8 or YOLOv9 (using ultralytics package)."""
    from ultralytics import YOLO
    
    # Load model
    model = YOLO(model_path)
    
    # Set task-specific parameters
    task_args = {'conf': confidence_threshold}
    if "Segmentation" in task_type:
        task_args['retina_masks'] = True
    
    # Run inference
    results = model(image_np, **task_args)
    
    # Process based on task type
    if task_type == "Object Detection":
        # Process detection results
        detections = []
        result = results[0]  # Get first result (only one image)
        
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = box.cls[0].item()
            class_id = int(cls)
            class_name = result.names[class_id]
            
            detections.append({
                'class_id': class_id,
                'class_name': class_name,
                'confidence': float(conf),
                'bbox': [float(x1), float(y1), float(x2), float(y2)]
            })
        
        # Get result image
        # First try the plot method
        try:
            # Plot with boxes
            rendered_img = results[0].plot()
            return Image.fromarray(rendered_img), detections
        except:
            # Fallback to manual rendering
            img = image_np.copy()
            for det in detections:
                x1, y1, x2, y2 = map(int, det['bbox'])
                label = f"{det['class_name']} {det['confidence']:.2f}"
                img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                img = cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Convert BGR to RGB if needed
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
            return Image.fromarray(img), detections
    
    elif task_type in ["Semantic Segmentation", "Instance Segmentation", "Panoptic Segmentation"]:
        # Process segmentation results
        segmentations = []
        result = results[0]  # Get first result
        
        # Check if masks are available
        if hasattr(result, 'masks') and result.masks is not None:
            for i, mask in enumerate(result.masks.data):
                # Get corresponding box
                box = result.boxes[i]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = box.cls[0].item()
                class_id = int(cls)
                class_name = result.names[class_id]
                
                segmentations.append({
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': float(conf),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'mask': mask.cpu().numpy()
                })
            
            # Try to use plot method for visualization
            try:
                rendered_img = result.plot()
                return Image.fromarray(rendered_img), segmentations
            except:
                # Fallback to manual rendering
                img = image_np.copy()
                for seg in segmentations:
                    mask = seg['mask'].astype(np.uint8) * 255
                    color_mask = np.zeros_like(img, dtype=np.uint8)
                    color = utils.get_color(seg['class_id'])
                    color_mask[mask > 0] = color
                    
                    # Blend mask with image
                    alpha = 0.5
                    img = cv2.addWeighted(img, 1, color_mask, alpha, 0)
                    
                    # Draw bounding box
                    x1, y1, x2, y2 = map(int, seg['bbox'])
                    label = f"{seg['class_name']} {seg['confidence']:.2f}"
                    img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    img = cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Convert BGR to RGB if needed
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                return Image.fromarray(img), segmentations
        else:
            # If no masks, fall back to detection
            return process_with_ultralytics(image_np, model_path, "Object Detection", confidence_threshold)
    
    # Default fallback - try to use plot method
    try:
        rendered_img = results[0].plot()
        return Image.fromarray(rendered_img), []
    except:
        # Just return the original image if all else fails
        return Image.fromarray(image_np), []
