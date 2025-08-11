import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import torch
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Load the trained YOLO model
@st.cache_resource
def load_model():
    try:
        model = YOLO('runs/train/exp2/weights/best.pt')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def process_image(image, model):
    try:
        # Make prediction
        results = model.predict(image, conf=0.15)
        
        # Get the first result
        result = results[0]
        
        # Convert PIL Image to OpenCV format
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Dictionary to store counts
        counts = {
            'carrot': {'organic': 0, 'inorganic': 0},
            'cucumber': {'organic': 0, 'inorganic': 0},
            'ladyfinger': {'organic': 0, 'inorganic': 0},
            'potato': {'organic': 0, 'inorganic': 0},
            'sweet_potato': {'organic': 0, 'inorganic': 0}
        }
        
        # Draw boxes and count detections
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            cls = int(box.cls[0])
            
            # Get class name
            class_name = result.names[cls]
            
            # Parse the class name
            parts = class_name.lower().split()
            status = parts[0]  # 'organic' or 'inorganic'
            
            # Handle 'sweet potato' special case
            if 'sweet' in parts and 'potato' in parts:
                veg_type = 'sweet_potato'
            else:
                veg_type = parts[-1]  # last word is the vegetable name
            
            # Update counts if the vegetable type exists in our dictionary
            if veg_type in counts:
                counts[veg_type][status] += 1
            
            # Draw rectangle
            color = (0, 255, 0) if status == 'organic' else (0, 0, 255)
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Add label with confidence
            display_veg = veg_type.replace('_', ' ')
            label = f"{display_veg} ({status}) {conf:.2f}"
            cv2.putText(img, label, (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Convert back to RGB for Streamlit
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, counts
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None

def main():
    try:
        st.set_page_config(page_title="Vegetable Detector", layout="wide")
        st.title("Vegetable Organic/Inorganic Detector")
        
        # Load model
        model = load_model()
        if model is None:
            st.error("Failed to load model")
            return
        
        # File uploader
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Read image
            image = Image.open(uploaded_file)
            
            # Create two columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)  # Updated parameter
            
            # Process image
            processed_img, counts = process_image(image, model)
            
            if processed_img is not None and counts is not None:
                with col2:
                    st.subheader("Detected Objects")
                    st.image(processed_img, use_container_width=True)  # Updated parameter
                
                # Display counts in a nice format
                st.subheader("Detection Summary")
                
                # Create table using st.columns
                for veg, status in counts.items():
                    if status['organic'] > 0 or status['inorganic'] > 0:
                        display_veg = veg.replace('_', ' ').title()
                        
                        # Create container for each vegetable
                        with st.container():
                            st.markdown(f"**{display_veg}:**")
                            cols = st.columns(2)
                            with cols[0]:
                                st.markdown(f"🌱 Organic: {status['organic']}")
                            with cols[1]:
                                st.markdown(f"🏭 Inorganic: {status['inorganic']}")
                            st.markdown("---")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()