import streamlit as st
import io
import json
import os
import numpy as np
from zipfile import ZipFile, ZIP_STORED
from PIL import Image
import pandas as pd # Added for timestamp

# Try to import PDF processing
try:
    from pdf2image import convert_from_bytes
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Import Augraphy components with better error handling
try:
    from augraphy import *
    # Test a basic Augraphy function
    from augraphy.augmentations.ink import InkBleed
    from augraphy.augmentations.paper import PaperFactory
    from augraphy.augmentations.geometric import Rotate
    AUGRAPHY_AVAILABLE = True
except ImportError as e:
    AUGRAPHY_AVAILABLE = False
    st.error(f"⚠️ Augraphy installation issue: {str(e)}")

# ---------------------- 1 · Streamlit App Configuration ----------------------
st.set_page_config(
    page_title="Document Fraud Simulator",
    page_icon="📄",
    layout="wide"
)

st.title("Document Fraud Simulation Lab")
st.markdown("**Transform genuine documents into realistic training datasets for fraud detection models**")

# Status grid with better feedback
status_cols = st.columns(4)

with status_cols[0]:
    if AUGRAPHY_AVAILABLE:
        st.success("Augraphy Ready")
        st.caption("Document augmentation")
    else:
        st.error("Augraphy Loading")
        st.caption("Installing dependencies...")

with status_cols[1]:
    if PDF_AVAILABLE:
        st.success("PDF Support")
        st.caption("PDF conversion ready")
    else:
        st.warning("PDF Limited")
        st.caption("Image processing only")

with status_cols[2]:
    st.success("Image Support")
    st.caption("PNG, JPEG, TIFF")

with status_cols[3]:
    st.info("Cloud Deployed")
    st.caption("Auto-updating")

# ---------------------- 2 · Helper Functions ----------------------
def load_pages(file) -> list[Image.Image]:
    """Load pages from uploaded file (images or PDFs)"""
    try:
        if file.type == "application/pdf" and PDF_AVAILABLE:
            # Convert PDF to images
            images = convert_from_bytes(file.read(), dpi=200)  # Lower DPI for cloud
            return images
        elif file.type == "application/pdf" and not PDF_AVAILABLE:
            st.error("📄 PDF processing not available. Please convert to image format first.")
            return []
        else:
            # Handle image files
            return [Image.open(file)]
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        return []

# Define distortion presets with fallbacks
def get_distortion_presets():
    """Get available distortion presets"""
    if not AUGRAPHY_AVAILABLE:
        return {
            "No Distortion": "Augraphy not available",
            "Basic Effects": "Augraphy not available"
        }
    
    try:
        return {
            "Light Aging": {
                "ink": [InkBleed(intensity_range=(0.1, 0.3))],
                "paper": [PaperFactory(paper_type="dirty", p=0.3)],
                "post": [Rotate(angle_range=(-1, 1))]
            },
            "Heavy Distortion": {
                "ink": [InkBleed(intensity_range=(0.4, 0.7))],
                "paper": [PaperFactory(paper_type="aged", p=0.7)],
                "post": [Rotate(angle_range=(-3, 3))]
            },
            "Scan Quality": {
                "ink": [InkBleed(intensity_range=(0.1, 0.2))],
                "paper": [PaperFactory(paper_type="clean", p=0.2)],
                "post": [Rotate(angle_range=(-0.5, 0.5))]
            }
        }
    except Exception as e:
        st.error(f"Error creating presets: {str(e)}")
        return {"No Effects": "Error in Augraphy setup"}

# ---------------------- 3 · Sidebar Controls ----------------------
with st.sidebar:
    st.header("Simulation Controls")
    
    # Upload section
    st.subheader("Training Samples")
    uploaded_files = st.file_uploader(
        "Upload genuine documents",
        type=['png', 'jpg', 'jpeg', 'tiff', 'pdf'],
        accept_multiple_files=True,
        help="Supports images and PDFs"
    )
    
    # Distortion settings
    st.subheader("Choose Distortion Type")
    presets = get_distortion_presets()
    selected_preset = st.selectbox(
        "Distortion preset",
        options=list(presets.keys()),
        help="Pre-configured distortion combinations"
    )
    
    # Generation settings
    st.subheader("Generation Settings")
    num_samples = st.slider(
        "Number of variations per document",
        min_value=1,
        max_value=10,  # Reduced for cloud
        value=3,
        help="Generate multiple variations of each document"
    )
    
    # Generate button
    generate_btn = st.button(
        "Generate Training Dataset",
        type="primary",
        disabled=not uploaded_files or not AUGRAPHY_AVAILABLE
    )

# ---------------------- 4 · Main Processing ----------------------
if uploaded_files and generate_btn:
    if not AUGRAPHY_AVAILABLE:
        st.error("❌ Augraphy is required for document processing. Please wait for installation to complete.")
        st.stop()
    
    try:
        # Create pipeline
        preset_config = presets[selected_preset]
        if isinstance(preset_config, str):
            st.error(f"❌ {preset_config}")
            st.stop()
        
        pipeline = AugraphyPipeline(
            ink_phase=preset_config.get("ink", []),
            paper_phase=preset_config.get("paper", []),
            post_phase=preset_config.get("post", [])
        )
        
        # Process files
        all_results = []
        progress_bar = st.progress(0)
        
        for file_idx, uploaded_file in enumerate(uploaded_files):
            st.write(f"🔄 Processing: {uploaded_file.name}")
            
            # Load document pages
            pages = load_pages(uploaded_file)
            if not pages:
                continue
            
            for page_idx, page in enumerate(pages):
                # Convert to numpy array
                img_array = np.array(page.convert('RGB'))
                
                # Generate variations
                for sample_idx in range(num_samples):
                    try:
                        # Apply augmentation
                        augmented = pipeline(img_array)
                        augmented_img = Image.fromarray(augmented)
                        
                        # Create metadata
                        metadata = {
                            "original_file": uploaded_file.name,
                            "page": page_idx + 1,
                            "variation": sample_idx + 1,
                            "preset": selected_preset,
                            "timestamp": str(pd.Timestamp.now())
                        }
                        
                        all_results.append({
                            "image": augmented_img,
                            "metadata": metadata,
                            "filename": f"{uploaded_file.name.split('.')[0]}_page{page_idx+1}_var{sample_idx+1}.png"
                        })
                        
                    except Exception as e:
                        st.warning(f"⚠️ Error processing variation {sample_idx+1}: {str(e)}")
            
            # Update progress
            progress_bar.progress((file_idx + 1) / len(uploaded_files))
        
        # Create downloadable ZIP
        if all_results:
            zip_buffer = io.BytesIO()
            with ZipFile(zip_buffer, 'w', ZIP_STORED) as zip_file:
                # Add images
                for result in all_results:
                    img_buffer = io.BytesIO()
                    result["image"].save(img_buffer, format='PNG')
                    zip_file.writestr(f"images/{result['filename']}", img_buffer.getvalue())
                
                # Add metadata
                metadata_json = json.dumps([r["metadata"] for r in all_results], indent=2)
                zip_file.writestr("metadata.json", metadata_json)
            
            zip_buffer.seek(0)
            
            # Download button
            st.success(f"✅ Generated {len(all_results)} augmented documents")
            st.download_button(
                label="Download Training Dataset",
                data=zip_buffer.getvalue(),
                file_name="fraud_training_dataset.zip",
                mime="application/zip"
            )
            
            # Show sample results
            st.subheader("Sample Results")
            cols = st.columns(min(3, len(all_results)))
            for i, result in enumerate(all_results[:3]):
                with cols[i]:
                    st.image(result["image"], caption=result["filename"], use_column_width=True)
        
    except Exception as e:
        st.error(f"❌ Processing error: {str(e)}")

# ---------------------- 5 · Metadata Information ----------------------
with st.expander("📊 About Metadata and Augmentation Tracking"):
    st.markdown("""
    ### Metadata Storage Format
    
    Each generated dataset includes a `metadata.json` file that tracks:
    
    ```json
    {
        "original_file": "document.pdf",
        "page": 1,
        "variation": 2,
        "preset": "Light Aging",
        "timestamp": "2024-01-15 10:30:45",
        "augmentations_applied": ["InkBleed", "PaperFactory", "Rotate"]
    }
    ```
    
    ### How to Use Metadata:
    
    1. **Training Labels**: Use `original_file` to create ground truth labels
    2. **Quality Control**: Track which augmentations work best
    3. **Dataset Analysis**: Analyze distribution of variations
    4. **Reproducibility**: Recreate specific augmentations
    
    ### Checking Metadata:
    
    ```python
    import json
    
    # Load metadata
    with open('metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Analyze augmentations
    for item in metadata:
        print(f"File: {item['original_file']} -> {item['preset']}")
    ```
    """)

# ---------------------- 6 · Troubleshooting ----------------------
if not AUGRAPHY_AVAILABLE:
    with st.expander("🔧 Troubleshooting Augraphy Installation"):
        st.markdown("""
        If Augraphy is not loading, this might be due to:
        
        1. **First deployment**: Cloud environments take time to install packages
        2. **Dependencies**: Some system libraries might be missing
        3. **Version conflicts**: Package version incompatibilities
        
        **Solutions:**
        
        - Wait 2-3 minutes for automatic installation
        - Refresh the page
        - Try uploading smaller files first
        
        **Local Alternative:**
        
        For full PDF support and guaranteed compatibility, 
        download the local version of this app.
        """) 