import os
from PIL import Image, ImageDraw, ImageFilter

def create_synthetic_cxr(output_path, is_abnormal=False):
    """
    Generates a synthetic, recognizable monochromatic image resembling a Chest X-ray.
    - Dark air-filled lungs (left/right).
    - Bright center representing spine and mediastinum.
    - Bright region for the cardiac silhouette.
    Uses pure PIL to avoid external dependencies.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 512x512 Grayscale image
    w, h = 512, 512
    img = Image.new("L", (w, h), 20)  # dark background
    draw = ImageDraw.Draw(img)
    
    # 1. Mediastinum & Spine (center vertical bright column)
    draw.polygon([(230, 20), (282, 20), (282, 500), (230, 500)], fill=95)
    
    # 2. Left Lung Field (anatomical right) - Darker region
    draw.ellipse([80, 60, 210, 420], fill=12)
    # Right Lung Field (anatomical left) - Darker region
    draw.ellipse([302, 60, 432, 420], fill=12)
    
    # 3. Ribs (horizontal-curved stripes)
    for y in range(80, 420, 40):
        # Left ribs
        draw.arc([60, y-20, 220, y+20], start=180, end=330, fill=45, width=4)
        # Right ribs
        draw.arc([292, y-20, 452, y+20], start=210, end=360, fill=45, width=4)
        
    # 4. Heart Silhouette (bright blob at bottom center-left)
    # Normal heart size: cardiothoracic ratio < 50%
    if is_abnormal:
        # Enlarged heart (cardiomegaly)
        draw.ellipse([200, 280, 360, 440], fill=145)
        # Add a patch of opacity in the right lower lung field (pneumonia signature)
        for i in range(10):
            draw.ellipse([320 + i*2, 320 + i*3, 380 - i, 370 - i], fill=65)
    else:
        # Normal heart
        draw.ellipse([215, 300, 315, 430], fill=125)
        
    # 5. Diaphragm dome at bottom
    draw.ellipse([40, 420, 230, 560], fill=55)
    draw.ellipse([280, 420, 470, 560], fill=55)
    
    # Apply a Gaussian blur filter to make it look organic
    final_img = img.filter(ImageFilter.GaussianBlur(radius=5))
    final_img.save(output_path)
    print(f"Generated synthetic CXR image: {output_path}")

if __name__ == "__main__":
    sample_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_images")
    create_synthetic_cxr(os.path.join(sample_dir, "normal_chest_xray.png"), is_abnormal=False)
    create_synthetic_cxr(os.path.join(sample_dir, "abnormal_chest_xray.png"), is_abnormal=True)
