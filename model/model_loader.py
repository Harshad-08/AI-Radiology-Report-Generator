import os
import sys

# Global variables to store the loaded model, processor, and system states
_model = None
_processor = None
_device = "cpu"
_is_mock_mode = False
_loaded_model_name = "Mock Inference Engine"

def detect_device():
    """
    Detects the best available compute device:
    1. CUDA (NVIDIA GPU)
    2. MPS (Apple Silicon GPU)
    3. CPU (Fallback)
    """
    global _device
    try:
        import torch
        if torch.cuda.is_available():
            _device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # Apple Silicon GPU support
            _device = "mps"
        else:
            _device = "cpu"
    except ImportError:
        _device = "cpu"
    return _device

def load_vlm_model():
    """
    Safe loader for Vision-Language Models.
    Tries models in order of priority:
    1. umarigan/blip-image-captioning-base-chestxray-finetuned
    2. Salesforce/blip-image-captioning-base
    
    If libraries are missing or loading fails (e.g. offline, OOM),
    it flags mock mode and continues gracefully.
    """
    global _model, _processor, _is_mock_mode, _loaded_model_name
    
    detect_device()
    print(f"System detected compute device: {_device.upper()}")
    
    try:
        # Check imports
        import torch
        from transformers import BlipProcessor, BlipForConditionalGeneration
    except ImportError as e:
        print(f"Warning: ML libraries could not be loaded ({str(e)}). Switching to local Mock Inference Engine.")
        _is_mock_mode = True
        return None, None, True
        
    model_candidates = [
        "umarigan/blip-image-captioning-base-chestxray-finetuned",
        "Salesforce/blip-image-captioning-base"
    ]
    
    # Check environment override (e.g., if user wants to force mock mode)
    if os.environ.get("FORCE_MOCK_MODEL", "false").lower() == "true":
        print("FORCE_MOCK_MODEL environment variable set to true. Activating Mock Mode.")
        _is_mock_mode = True
        return None, None, True

    for model_name in model_candidates:
        try:
            print(f"Attempting to download and load VLM: '{model_name}' on {_device.upper()}...")
            # Load processor
            processor = BlipProcessor.from_pretrained(model_name, local_files_only=False)
            # Load model
            model = BlipForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float32)
            model.to(_device)
            model.eval()  # Set to evaluation mode
            
            _model = model
            _processor = processor
            _is_mock_mode = False
            _loaded_model_name = model_name
            print(f"Successfully loaded model '{model_name}' on device {_device.upper()}.")
            return _model, _processor, _is_mock_mode
            
        except Exception as e:
            print(f"Failed to load '{model_name}' due to error: {str(e)}")
            print("Retrying with fallback candidate...")
            
    print("All Hugging Face model loading attempts failed (No internet connection or OOM). Enabling high-fidelity Mock Mode.")
    _is_mock_mode = True
    return None, None, True

def get_loaded_model_details():
    """
    Returns diagnostic details about the currently loaded VLM.
    """
    return {
        "device": _device,
        "is_mock": _is_mock_mode,
        "model_name": _loaded_model_name
    }
