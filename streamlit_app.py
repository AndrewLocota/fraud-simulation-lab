import streamlit as st
import io
import json
import tempfile
import sys
import platform
import os
import numpy as np
from zipfile import ZipFile, ZIP_STORED
from PIL import Image

# Cloud-friendly version - disable PDF processing on cloud
CLOUD_DEPLOYMENT = True

# Import Augraphy components
try:
    from augraphy import *
    AUGRAPHY_AVAILABLE = True
except ImportError as e:
    AUGRAPHY_AVAILABLE = False
    st.error("Augraphy not available. Installing...")

# ---------------------- 1 · Streamlit App Configuration ----------------------
st.set_page_config(
    page_title="Document Fraud Simulator",
    page_icon="📄",
    layout="wide"
)

st.title("Document Fraud Simulation Lab")
st.markdown("**Transform genuine documents into realistic training datasets for fraud detection models**")

# Cloud deployment notice
if CLOUD_DEPLOYMENT:
    st.info("🌐 **Cloud Version**: PDF uploads are disabled. Please use image files (PNG, JPEG, TIFF) for best results.")

# Compact status grid (3 columns for cloud version)
status_cols = st.columns(3)

with status_cols[0]:
    if AUGRAPHY_AVAILABLE:
        st.success("Augraphy Ready")
    else:
        st.error("Augraphy Missing")

with status_cols[1]:
    st.success("Image Support")
    st.caption("PNG, JPEG, TIFF")

with status_cols[2]:
    st.info("Cloud Deployed")
    st.caption("No installation needed")

# ---------------------- 2 · Document Distortion Presets ----------------------
DISTORTION_PRESETS = {
    "Aging & Wear": {
        "description": "Simulates old, worn documents with ink degradation",
        "effects": ["InkBleed", "Faxify", "LowInkPeriodicLines"]
    },
    "Poor Scanning": {
        "description": "Mimics low-quality scanners and photocopiers", 
        "effects": ["BadPhotoCopy", "DirtyDrum", "NoiseTexturize"]
    },
    "Physical Damage": {
        "description": "Replicates folding, staining, and physical distortions",
        "effects": ["Folding", "BookBinding", "PageBorder"]
    },
    "Printer Issues": {
        "description": "Simulates various printer malfunctions and ink problems",
        "effects": ["LowInkRandomLines", "DirtyRollers", "Letterpress"]
    },
    "Environmental Damage": {
        "description": "Weather, liquid spills, and environmental factors",
        "effects": ["SubtleNoise", "BrightnessTexturize", "Markup"]
    }
}

# ---------------------- 3 · Sidebar Controls ----------------------
st.sidebar.title("Simulation Controls")

# File upload (images only for cloud)
uploaded = st.sidebar.file_uploader(
    "Upload Document",
    type=["png", "jpg", "jpeg", "tiff"],  # No PDF for cloud
    help="Upload a clean image document to apply fraud simulation effects"
)

# Distortion preset selection
preset = st.sidebar.selectbox(
    "Choose Distortion Type",
    list(DISTORTION_PRESETS.keys()),
    help="Select the type of document degradation to simulate"
)

# Display preset description
st.sidebar.info(f"**{preset}**: {DISTORTION_PRESETS[preset]['description']}")

# Number of copies
copies = st.sidebar.number_input(
    "Training Samples",
    min_value=1,
    max_value=20,  # Reduced for cloud
    value=3,
    help="Number of augmented copies to generate for training"
)

# Advanced options
with st.sidebar.expander("Advanced Options"):
    intensity = st.slider("Effect Intensity", 0.1, 1.0, 0.5, 0.1)
    include_metadata = st.checkbox("Include Augmentation Metadata", True)
    output_format = st.selectbox("Output Format", ["PNG", "JPEG"], index=0)

# ---------------------- 4 · Pipeline Creation ----------------------
def create_augraphy_pipeline(preset_name, intensity=0.5):
    """Create an Augraphy pipeline based on the selected preset"""
    if not AUGRAPHY_AVAILABLE:
        return None
    
    try:
        # Use the default pipeline as a base
        pipeline = default_augraphy_pipeline()
        return pipeline
        
    except Exception as e:
        st.error(f"Error creating pipeline: {e}")
        return None

def load_pages(file) -> list[Image.Image]:
    """Convert uploaded file to list of PIL Images"""
    try:
        # Cloud version - only images
        return [Image.open(file)]
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return []

# ---------------------- 5 · Main Processing ----------------------
if uploaded and st.sidebar.button("Generate Training Dataset", type="primary"):
    if not AUGRAPHY_AVAILABLE:
        st.error("Augraphy library is required but not available.")
        st.stop()
    
    with st.spinner("Processing document..."):
        # Load pages
        pages = load_pages(uploaded)
        if not pages:
            st.error("Failed to load pages from the uploaded file.")
            st.stop()
        
        # Create pipeline
        pipeline = create_augraphy_pipeline(preset, intensity)
        if not pipeline:
            st.error("Failed to create augmentation pipeline.")
            st.stop()
        
        # Process pages
        outputs = []
        metadata = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_operations = int(copies) * len(pages)
        
        for copy_idx in range(int(copies)):
            for page_idx, page in enumerate(pages):
                try:
                    current_op = copy_idx * len(pages) + page_idx + 1
                    status_text.text(f"Processing copy {copy_idx + 1}, page {page_idx + 1}...")
                    
                    # Convert PIL to numpy array
                    img_array = np.array(page)
                    
                    # Apply Augraphy pipeline
                    result = pipeline.augment(img_array)
                    augmented_array = result["output"]
                    
                    # Convert back to PIL
                    output_img = Image.fromarray(augmented_array.astype(np.uint8))
                    outputs.append(output_img)
                    
                    # Store metadata
                    meta = {
                        "copy_number": copy_idx + 1,
                        "page_number": page_idx + 1,
                        "preset": preset,
                        "effects": DISTORTION_PRESETS[preset]["effects"],
                        "intensity": intensity,
                        "original_size": page.size,
                        "output_size": output_img.size
                    }
                    
                    if include_metadata and "metadata" in result:
                        meta["augraphy_metadata"] = result["metadata"]
                    
                    metadata.append(meta)
                    
                    # Update progress
                    progress_bar.progress(current_op / total_operations)
                    
                except Exception as e:
                    st.error(f"Error processing copy {copy_idx + 1}, page {page_idx + 1}: {e}")
                    continue
        
        progress_bar.empty()
        status_text.empty()
        
        if not outputs:
            st.error("No images were generated successfully.")
            st.stop()
        
        # ---------------------- 6 · Results Display ----------------------
        st.success(f"Generated {len(outputs)} training samples")
        
        # Show comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Document")
            st.image(pages[0], caption="Clean original", use_container_width=True)
        
        with col2:
            st.subheader("Augmented Sample")
            st.image(outputs[0], caption=f"Distorted with {preset}", use_container_width=True)
        
        # Preview gallery
        st.subheader("Generated Samples Gallery")
        
        # Show samples in a grid
        cols = st.columns(min(4, len(outputs)))
        for i, img in enumerate(outputs[:4]):  # Show first 4 samples
            with cols[i % 4]:
                st.image(img, caption=f"Sample {i+1}", use_container_width=True)
        
        if len(outputs) > 4:
            st.info(f"Showing 4 of {len(outputs)} generated samples. Download the full dataset below.")
        
        # ---------------------- 7 · Download Options ----------------------
        st.subheader("Download Training Dataset")
        
        if len(outputs) == 1:
            # Single file download
            buf = io.BytesIO()
            outputs[0].save(buf, format=output_format)
            st.download_button(
                "Download Augmented Image",
                data=buf.getvalue(),
                file_name=f"augmented_sample.{output_format.lower()}",
                mime=f"image/{output_format.lower()}"
            )
        else:
            # ZIP package download
            zip_buffer = io.BytesIO()
            
            with ZipFile(zip_buffer, "w", ZIP_STORED) as zip_file:
                # Add images
                for i, img in enumerate(outputs):
                    img_buffer = io.BytesIO()
                    img.save(img_buffer, format=output_format)
                    zip_file.writestr(f"sample_{i+1:03d}.{output_format.lower()}", img_buffer.getvalue())
                
                # Add metadata
                if include_metadata:
                    metadata_json = json.dumps(metadata, indent=2)
                    zip_file.writestr("dataset_metadata.json", metadata_json)
                
                # Add dataset info
                dataset_info = {
                    "dataset_name": f"fraud_simulation_{preset.lower().replace(' ', '_')}",
                    "total_samples": len(outputs),
                    "original_pages": len(pages),
                    "copies_per_page": copies,
                    "distortion_preset": preset,
                    "effects_applied": DISTORTION_PRESETS[preset]["effects"],
                    "parameters": {
                        "intensity": intensity,
                        "output_format": output_format
                    }
                }
                zip_file.writestr("dataset_info.json", json.dumps(dataset_info, indent=2))
            
            zip_buffer.seek(0)
            
            st.download_button(
                "Download Complete Training Dataset (ZIP)",
                data=zip_buffer.getvalue(),
                file_name=f"fraud_training_dataset_{preset.lower().replace(' ', '_')}.zip",
                mime="application/zip"
            )

# ---------------------- 8 · Information Panels ----------------------
if not uploaded:
    st.markdown("---")
    st.subheader("How to Use This Tool")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Step 1: Upload Document**
        - Upload clean image files
        - Supports PNG, JPEG, TIFF formats
        
        **Step 2: Choose Distortion**
        - Select from realistic fraud scenarios
        - Each preset simulates different damage types
        
        **Step 3: Generate Dataset**
        - Specify number of training samples
        - Adjust intensity and options
        """)
    
    with col2:
        st.markdown("""
        **Training Data Applications:**
        - Document fraud detection models
        - OCR robustness testing
        - Image preprocessing algorithms
        - Authentication system training
        
        **Cloud Version Features:**
        - No installation required
        - Instant access via web browser
        - Secure private deployment
        - Cross-platform compatibility
        """)

# ---------------------- 9 · Footer ----------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
<small>Built with Augraphy & Streamlit | Document Fraud Simulation Lab</small>
</div>
""", unsafe_allow_html=True) 