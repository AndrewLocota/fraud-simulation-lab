import streamlit as st
import io
import json
import os
import numpy as np
from zipfile import ZipFile, ZIP_STORED
from PIL import Image

# Set OpenCV to headless mode before importing anything that uses it
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '0'

# Try to import PDF processing
try:
    from pdf2image import convert_from_bytes
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Import Augraphy with better error handling for cloud deployment
AUGRAPHY_AVAILABLE = False
try:
    # Try to import with minimal dependencies first
    import cv2
    cv2.setNumThreads(1)  # Reduce threading for cloud
    
    # Now try Augraphy
    from augraphy import AugraphyPipeline
    from augraphy.augmentations.ink import InkBleed, Fading
    from augraphy.augmentations.paper import BadPhotoCopy, DirtyDrum
    from augraphy.augmentations.geometric import Rotate
    
    AUGRAPHY_AVAILABLE = True
    AUGRAPHY_ERROR = None
    
except ImportError as e:
    AUGRAPHY_ERROR = str(e)
    st.warning(f"🔧 Augraphy loading issue: {AUGRAPHY_ERROR}")
    st.info("💡 **Workaround**: Using basic PIL effects instead")

except Exception as e:
    AUGRAPHY_ERROR = str(e)
    st.error(f"⚠️ System compatibility issue: {AUGRAPHY_ERROR}")

# ---------------------- Fallback Effects (PIL-based) ----------------------
def apply_basic_distortion(image, effect_type="light"):
    """Apply basic distortions using PIL when Augraphy isn't available"""
    from PIL import ImageEnhance, ImageFilter
    import random
    
    try:
        if effect_type == "light":
            # Light aging effect
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(random.uniform(0.9, 1.1))
            
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.95, 1.05))
            
        elif effect_type == "heavy":
            # Heavy distortion
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(random.uniform(0.7, 1.3))
            
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))
            
            # Add slight blur
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 0.5)))
            
        elif effect_type == "scan":
            # Scan-like quality
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(random.uniform(0.8, 1.2))
            
        return image
    except Exception as e:
        st.warning(f"Basic effect error: {e}")
        return image

# ---------------------- 1 · Streamlit App Configuration ----------------------
st.set_page_config(
    page_title="Document Fraud Simulator",
    page_icon="📄",
    layout="wide"
)

st.title("Document Fraud Simulation Lab")
st.markdown("**Transform genuine documents into realistic training datasets for fraud detection models**")

# Status grid with current capabilities
status_cols = st.columns(4)

with status_cols[0]:
    if AUGRAPHY_AVAILABLE:
        st.success("Augraphy Ready")
        st.caption("Full augmentation")
    else:
        st.warning("Basic Effects")
        st.caption("PIL-based distortions")

with status_cols[1]:
    if PDF_AVAILABLE:
        st.success("PDF Support")
        st.caption("PDF conversion ready")
    else:
        st.warning("Images Only")
        st.caption("PNG, JPEG, TIFF")

with status_cols[2]:
    st.success("Cloud Ready")
    st.caption("No installation needed")

with status_cols[3]:
    st.info("Auto-Deploy")
    st.caption("Updates automatically")

# ---------------------- 2 · Helper Functions ----------------------
def load_pages(file) -> list[Image.Image]:
    """Load pages from uploaded file"""
    try:
        if file.type == "application/pdf" and PDF_AVAILABLE:
            images = convert_from_bytes(file.read(), dpi=150)
            return images
        elif file.type == "application/pdf":
            st.error("📄 PDF processing not available. Please use image files.")
            return []
        else:
            return [Image.open(file)]
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        return []

def get_distortion_presets():
    """Get available distortion presets based on what's available"""
    if AUGRAPHY_AVAILABLE:
        try:
            return {
                "Light Aging": {
                    "type": "augraphy",
                    "ink": [InkBleed(intensity_range=(0.1, 0.3))],
                    "paper": [BadPhotoCopy(noise_type=1, noise_side="random")],
                    "post": [Rotate(angle_range=(-1, 1))]
                },
                "Heavy Distortion": {
                    "type": "augraphy", 
                    "ink": [InkBleed(intensity_range=(0.4, 0.7)), Fading(fading_value_range=(0.1, 0.3))],
                    "paper": [DirtyDrum(line_width_range=(1, 6))],
                    "post": [Rotate(angle_range=(-3, 3))]
                },
                "Document Scan": {
                    "type": "augraphy",
                    "ink": [InkBleed(intensity_range=(0.05, 0.15))],
                    "paper": [BadPhotoCopy(noise_type=0, noise_side="random")],
                    "post": [Rotate(angle_range=(-0.5, 0.5))]
                }
            }
        except Exception as e:
            st.warning(f"Augraphy preset error: {e}")
            return get_basic_presets()
    else:
        return get_basic_presets()

def get_basic_presets():
    """Fallback presets using PIL"""
    return {
        "Light Effects": {"type": "basic", "effect": "light"},
        "Heavy Effects": {"type": "basic", "effect": "heavy"}, 
        "Scan Quality": {"type": "basic", "effect": "scan"}
    }

# ---------------------- 3 · Sidebar Controls ----------------------
with st.sidebar:
    st.header("Simulation Controls")
    
    # Show current mode
    if AUGRAPHY_AVAILABLE:
        st.success("🎯 **Advanced Mode**: Full Augraphy effects available")
    else:
        st.info("🔧 **Basic Mode**: Using PIL-based effects")
        with st.expander("Why Basic Mode?"):
            st.markdown(f"""
            **Issue**: {AUGRAPHY_ERROR if AUGRAPHY_ERROR else 'Augraphy dependencies not available'}
            
            **Current Capabilities**:
            - ✅ Contrast/brightness adjustments
            - ✅ Blur and sharpness effects  
            - ✅ Basic aging simulation
            - ❌ Advanced ink/paper effects
            
            **For full features**: Download the local version
            """)
    
    st.subheader("Training Samples")
    uploaded_files = st.file_uploader(
        "Upload genuine documents",
        type=['png', 'jpg', 'jpeg', 'tiff'] + (['pdf'] if PDF_AVAILABLE else []),
        accept_multiple_files=True
    )
    
    st.subheader("Choose Distortion Type")
    presets = get_distortion_presets()
    selected_preset = st.selectbox(
        "Distortion preset",
        options=list(presets.keys())
    )
    
    st.subheader("Generation Settings")
    num_samples = st.slider(
        "Variations per document",
        min_value=1,
        max_value=8,
        value=3
    )
    
    generate_btn = st.button(
        "Generate Training Dataset",
        type="primary",
        disabled=not uploaded_files
    )

# ---------------------- 4 · Main Processing ----------------------
if uploaded_files and generate_btn:
    try:
        preset_config = presets[selected_preset]
        all_results = []
        progress_bar = st.progress(0)
        
        for file_idx, uploaded_file in enumerate(uploaded_files):
            st.write(f"🔄 Processing: {uploaded_file.name}")
            
            pages = load_pages(uploaded_file)
            if not pages:
                continue
            
            for page_idx, page in enumerate(pages):
                for sample_idx in range(num_samples):
                    try:
                        if preset_config["type"] == "augraphy" and AUGRAPHY_AVAILABLE:
                            # Use Augraphy pipeline
                            pipeline = AugraphyPipeline(
                                ink_phase=preset_config.get("ink", []),
                                paper_phase=preset_config.get("paper", []),
                                post_phase=preset_config.get("post", [])
                            )
                            img_array = np.array(page.convert('RGB'))
                            augmented_array = pipeline(img_array)
                            augmented_img = Image.fromarray(augmented_array)
                        else:
                            # Use basic PIL effects
                            augmented_img = apply_basic_distortion(page, preset_config["effect"])
                        
                        # Create metadata
                        metadata = {
                            "original_file": uploaded_file.name,
                            "page": page_idx + 1,
                            "variation": sample_idx + 1,
                            "preset": selected_preset,
                            "method": preset_config["type"],
                            "timestamp": str(np.datetime64('now'))
                        }
                        
                        all_results.append({
                            "image": augmented_img,
                            "metadata": metadata,
                            "filename": f"{uploaded_file.name.split('.')[0]}_p{page_idx+1}_v{sample_idx+1}.png"
                        })
                        
                    except Exception as e:
                        st.warning(f"⚠️ Error in variation {sample_idx+1}: {str(e)}")
            
            progress_bar.progress((file_idx + 1) / len(uploaded_files))
        
        # Create download
        if all_results:
            zip_buffer = io.BytesIO()
            with ZipFile(zip_buffer, 'w', ZIP_STORED) as zip_file:
                for result in all_results:
                    img_buffer = io.BytesIO()
                    result["image"].save(img_buffer, format='PNG')
                    zip_file.writestr(f"images/{result['filename']}", img_buffer.getvalue())
                
                metadata_json = json.dumps([r["metadata"] for r in all_results], indent=2)
                zip_file.writestr("metadata.json", metadata_json)
            
            zip_buffer.seek(0)
            
            st.success(f"✅ Generated {len(all_results)} augmented documents")
            st.download_button(
                label="📥 Download Training Dataset",
                data=zip_buffer.getvalue(),
                file_name="fraud_training_dataset.zip",
                mime="application/zip"
            )
            
            # Show samples
            st.subheader("Sample Results")
            cols = st.columns(min(3, len(all_results)))
            for i, result in enumerate(all_results[:3]):
                with cols[i]:
                    st.image(result["image"], caption=result["filename"], use_column_width=True)
        
    except Exception as e:
        st.error(f"❌ Processing error: {str(e)}")

# ---------------------- 5 · Information ----------------------
with st.expander("📊 About This Tool"):
    st.markdown("""
    ### Cloud vs Local Version
    
    **Cloud Version (Current)**:
    - ✅ No installation required
    - ✅ Basic document distortion effects
    - ✅ Automatic updates
    - ❌ Limited to basic PIL effects if Augraphy fails
    
    **Local Version**:
    - ✅ Full Augraphy effects library
    - ✅ PDF processing with Poppler
    - ✅ Higher sample limits
    - ✅ Advanced ink/paper/geometric effects
    
    ### Metadata Format
    Each dataset includes tracking information for reproducibility and analysis.
    """) 