import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def get_color(class_id):
    """Generate a color for a class ID."""
    np.random.seed(class_id)
    color = tuple(map(int, np.random.randint(0, 255, size=3)))
    return color

def show_detection_details(results):
    """Display detailed detection results in a Streamlit app."""
    if not results:
        st.write("No detections found")
        return
    
    # Prepare data for display
    data = []
    for i, result in enumerate(results):
        data.append({
            "ID": i + 1,
            "Class": result.get('class_name', ''),
            "Confidence": f"{result.get('confidence', 0):.2f}",
            "Bounding Box": [int(x) for x in result.get('bbox', [0, 0, 0, 0])],
        })
    
    # Create and display dataframe
    df = pd.DataFrame(data)
    st.dataframe(df)
    
    # Class distribution chart
    if len(results) > 0:
        st.subheader("Class Distribution")
        class_counts = {}
        for result in results:
            class_name = result.get('class_name', 'Unknown')
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        fig, ax = plt.subplots()
        ax.bar(class_counts.keys(), class_counts.values())
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_title('Object Class Distribution')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        
    # Confidence distribution
    if len(results) > 0:
        confidences = [result.get('confidence', 0) for result in results]
        
        st.subheader("Confidence Distribution")
        fig, ax = plt.subplots()
        ax.hist(confidences, bins=10, range=(0, 1))
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Count')
        ax.set_title('Confidence Score Distribution')
        st.pyplot(fig)

def render_bounding_boxes(image, detections):
    """Render bounding boxes on an image."""
    import cv2
    
    img = np.array(image).copy()
    
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        color = get_color(det['class_id'])
        label = f"{det['class_name']} {det['confidence']:.2f}"
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(img, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
        
        # Draw label text
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return img

def render_masks(image, segmentations):
    """Render segmentation masks on an image."""
    import cv2
    
    img = np.array(image).copy()
    overlay = img.copy()
    
    for seg in segmentations:
        if 'mask' not in seg:
            continue
            
        mask = seg['mask']
        if len(mask.shape) == 3:
            # Multiple masks per instance, combine them
            mask = np.any(mask, axis=0).astype(np.uint8)
        
        # Resize mask if needed
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (img.shape[1], img.shape[0]))
        
        # Create colored mask
        color = get_color(seg['class_id'])
        colored_mask = np.zeros_like(img)
        colored_mask[mask > 0] = color
        
        # Blend mask with image
        alpha = 0.5
        cv2.addWeighted(overlay, 1 - alpha, colored_mask, alpha, 0, overlay)
        
        # Add bounding box
        x1, y1, x2, y2 = map(int, seg['bbox'])
        label = f"{seg['class_name']} {seg['confidence']:.2f}"
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(overlay, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return overlay

def format_processing_time(seconds):
    """Format processing time nicely."""
    if seconds < 0.001:
        return f"{seconds * 1000000:.2f} µs"
    elif seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    else:
        return f"{seconds:.2f} s"
