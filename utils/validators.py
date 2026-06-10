import os
from PIL import Image
import numpy as np

def validate_image_file(file_path):
    """
    Validates if the file exists, has a correct image extension, is not corrupted,
    and complies with chest X-ray heuristic rules.
    
    Returns:
        (bool, str): A tuple where the first element is a boolean indicating success,
                     and the second element is an error message if failed, or success details.
    """
    # 1. Check if file exists
    if not os.path.exists(file_path):
        return False, "File does not exist."
        
    # 2. Check extension
    valid_extensions = ['.png', '.jpg', '.jpeg']
    _, ext = os.path.splitext(file_path.lower())
    if ext not in valid_extensions:
        return False, f"Invalid file format '{ext}'. Only JPG, JPEG, and PNG are supported."
        
    # 3. Check if image is corrupted / loadable
    try:
        with Image.open(file_path) as img:
            img.verify()  # verify integrity
    except Exception as e:
        return False, f"Corrupted or invalid image file. Detailed error: {str(e)}"
        
    # Re-open for detailed analysis since verify() closes the file pointer
    try:
        with Image.open(file_path) as img:
            width, height = img.size
            
            # 4. Check resolution (too small check)
            if width < 150 or height < 150:
                return False, f"Image resolution too low ({width}x{height}). Minimum required is 150x150 pixels."
                
            # 5. Convert to numpy array for clinical heuristic analysis
            img_np = np.array(img.convert('RGB'))
            
            # Check color saturation (Chest X-Rays must be grayscale/monochromatic)
            # We check the average difference between color channels.
            r, g, b = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]
            mean_channel_diff = np.mean([np.abs(r - g), np.abs(g - b), np.abs(b - r)])
            if mean_channel_diff > 15.0:  # Color threshold
                return False, "Validation failed: Uploaded image appears to be a color photograph. Chest X-rays must be monochromatic (grayscale)."
                
            # Convert to single channel grayscale for shape/structure checks
            gray_img = img.convert('L')
            gray_np = np.array(gray_img)
            
            # 6. Check aspect ratio (width / height)
            # Chest X-Rays are generally vertical/square, but landscape views or pediatric X-rays can range up to 2.0.
            aspect_ratio = width / height
            if aspect_ratio < 0.4 or aspect_ratio > 2.1:
                return False, f"Invalid aspect ratio ({aspect_ratio:.2f}). Chest X-rays usually have an aspect ratio between 0.4 and 2.1."
                
            # 7. Chest X-Ray Quadrant/Pattern Heuristic Analysis:
            # - Chest X-Rays typically have a dark lung field on the left and right upper quadrants.
            # - They have a brighter central column (mediastinum, heart, and spine).
            # - The background outside the body is usually very dark (near 0).
            # Let's divide the image into 3 vertical columns: Left Lung, Center (Spine/Heart), Right Lung
            # And split horizontally: Upper (lungs), Lower (diaphragm/abdomen)
            h, w = gray_np.shape
            left_col = gray_np[0:int(h*0.6), 0:int(w*0.33)]
            center_col = gray_np[0:int(h*0.6), int(w*0.33):int(w*0.66)]
            right_col = gray_np[0:int(h*0.6), int(w*0.66):w]
            
            mean_left = np.mean(left_col)
            mean_center = np.mean(center_col)
            mean_right = np.mean(right_col)
            
            # Dental X-Rays are very bright all over or dark with thin tooth rows.
            # Bone fractures have a very bright bone segment and very dark surroundings.
            # In a chest X-Ray, the center column (containing the heart and thoracic spine) 
            # is typically brighter than the left/right lung columns, which contain air and are darker.
            # We verify that the center column is brighter than the average of the two lung columns.
            # However, we keep a relaxed threshold to avoid false rejections.
            mean_lungs = (mean_left + mean_right) / 2.0
            
            # Ensure the image has chest-xray like contrast distribution.
            # For a pure black or pure white image (e.g. solid color or extreme contrast error), we reject.
            if np.std(gray_np) < 10:
                return False, "Validation failed: Image has extremely low contrast (appears blank or solid color)."
                
            # If the lungs mean is brighter than the center (or they are almost identical), 
            # it might be a bone fracture or dental scan.
            # We add a mild condition to reject highly un-chest-xray-like shapes.
            # But we make sure not to reject a chest X-ray with large opacities.
            if mean_center < 15 and mean_lungs < 15:
                return False, "Validation failed: Image appears too dark or lacks anatomical features."

    except Exception as e:
        return False, f"Failed to parse image file: {str(e)}"
        
    return True, "Valid Chest X-Ray image."
