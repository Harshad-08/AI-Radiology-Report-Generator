from PIL import Image

def preprocess_chest_xray(image_path, target_size=(384, 384)):
    """
    Standardizes the input chest X-ray image:
    1. Loads the image from path.
    2. Converts grayscale or RGBA to RGB mode (3 channels).
    3. Resizes to the model target dimensions (default 384x384 for BLIP).
    
    Returns:
        PIL.Image: Preprocessed PIL Image.
    """
    try:
        # Load the image
        img = Image.open(image_path)
        
        # Convert to RGB mode if not already (BLIP requires 3 channels)
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Resize to target size for the VLM model input
        if target_size:
            img = img.resize(target_size, Image.Resampling.BILINEAR)
            
        return img
    except Exception as e:
        raise ValueError(f"Error during image preprocessing: {str(e)}")
