import os
import subprocess
import sys
import importlib.util
from pathlib import Path

# Model directory paths
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Dictionary mapping YOLO versions to their package and model prefix
YOLO_VERSIONS = {
    "YOLOv5": {
        "package": "yolov5",
        "model_prefix": "yolov5",
        "module": "models"
    },
    "YOLOv8": {
        "package": "ultralytics",
        "model_prefix": "yolov8",
        "module": "ultralytics.models"
    },
    "YOLOv9": {
        "package": "ultralytics",  # YOLOv9 is also available in ultralytics package
        "model_prefix": "yolov9",
        "module": "ultralytics.models"
    },
    "YOLOv11": {  # Adding YOLOv11 (future version)
        "package": "ultralytics",
        "model_prefix": "yolov11",
        "module": "ultralytics.models"
    }
}

# Task mapping
TASK_MAP = {
    "Object Detection": "detect",
    "Semantic Segmentation": "segment",
    "Instance Segmentation": "segment",  # YOLOv8 uses same model for both segmentations
    "Panoptic Segmentation": "segment-p"  # Custom suffix for panoptic segmentation
}

# Check for torch availability without importing
torch_available = importlib.util.find_spec("torch") is not None

# Model sizes
MODEL_SIZES = ["n", "s", "m", "l", "x"]

def check_package_installed(package_name):
    """Check if a Python package is installed."""
    return importlib.util.find_spec(package_name) is not None

def ensure_package_installed(package_name):
    """Ensure a Python package is installed."""
    if not check_package_installed(package_name):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            return True
        except subprocess.CalledProcessError:
            return False
    return True

def get_available_models():
    """Get a list of all available model files."""
    models = []
    
    # Check for installed packages first
    for version, info in YOLO_VERSIONS.items():
        if check_package_installed(info["package"]):
            # Try to get models from the package
            try:
                if version == "YOLOv5" and check_package_installed("yolov5"):
                    import yolov5
                    model_dir = Path(yolov5.__file__).parent / "models"
                    pt_files = list(model_dir.glob("*.pt"))
                    for pt_file in pt_files:
                        models.append(f"{version}/{pt_file.name}")
                elif check_package_installed("ultralytics"):  # YOLOv8 and YOLOv9
                    # Add default model names based on convention
                    prefix = info["model_prefix"]
                    for size in MODEL_SIZES:
                        for task_key, task_suffix in TASK_MAP.items():
                            # Only add if it matches the expected pattern
                            if task_suffix == "detect":
                                models.append(f"{version}/{prefix}{size}.pt")
                            elif task_suffix == "segment":
                                models.append(f"{version}/{prefix}{size}-seg.pt")
                            elif task_suffix == "segment-p":
                                # YOLOv8 has panoptic segmentation models
                                if version == "YOLOv8":
                                    models.append(f"{version}/{prefix}{size}-p.pt")
            except Exception as e:
                print(f"Error getting models for {version}: {e}")
                pass
    
    # Local models directory
    if MODEL_DIR.exists():
        for pt_file in MODEL_DIR.glob("*.pt"):
            is_v5 = "yolov5" in pt_file.name.lower()
            is_v8 = "yolov8" in pt_file.name.lower()
            is_v9 = "yolov9" in pt_file.name.lower()
            
            if is_v5:
                models.append(f"YOLOv5/{pt_file.name}")
            elif is_v8:
                models.append(f"YOLOv8/{pt_file.name}")
            elif is_v9:
                models.append(f"YOLOv9/{pt_file.name}")
            else:
                # Generic model, try to guess version based on file name
                if "yolo" in pt_file.name.lower():
                    models.append(f"YOLOv5/{pt_file.name}")  # Default to YOLOv5
    
    return models

def get_model_names(version, task_type):
    """Get available model names for the selected version and task."""
    task_suffix = TASK_MAP.get(task_type, "detect")
    prefix = YOLO_VERSIONS[version]["model_prefix"]
    
    available_models = get_available_models()
    filtered_models = []
    
    for model in available_models:
        if not model.startswith(version + "/"):
            continue
            
        model_name = model.split("/")[1]
        
        # Filter based on task type
        if task_type == "Object Detection" and "-seg" not in model_name and "-p" not in model_name:
            filtered_models.append(model_name)
        elif task_type in ["Semantic Segmentation", "Instance Segmentation"] and "-seg" in model_name:
            filtered_models.append(model_name)
        elif task_type == "Panoptic Segmentation" and "-p" in model_name:
            filtered_models.append(model_name)
    
    return filtered_models

def get_downloadable_models(version, task_type):
    """Get a list of models that can be downloaded."""
    task_suffix = TASK_MAP.get(task_type, "detect")
    prefix = YOLO_VERSIONS[version]["model_prefix"]
    
    available_models = get_available_models()
    available_model_names = [model.split("/")[1] for model in available_models if model.startswith(version + "/")]
    
    downloadable_models = []
    
    # Generate potential model names
    for size in MODEL_SIZES:
        if task_type == "Object Detection":
            model_name = f"{prefix}{size}.pt"
            if model_name not in available_model_names:
                downloadable_models.append(model_name)
        
        elif task_type in ["Semantic Segmentation", "Instance Segmentation"]:
            model_name = f"{prefix}{size}-seg.pt"
            if model_name not in available_model_names:
                downloadable_models.append(model_name)
        
        elif task_type == "Panoptic Segmentation" and version == "YOLOv8":
            model_name = f"{prefix}{size}-p.pt"
            if model_name not in available_model_names:
                downloadable_models.append(model_name)
    
    return downloadable_models

def download_model(version, model_name, task_type):
    """Download the specified model."""
    # Ensure required packages are installed
    package_name = YOLO_VERSIONS[version]["package"]
    package_installed = ensure_package_installed(package_name)
    
    if not package_installed:
        return False
    
    # Check if torch is available
    if not torch_available:
        return False
    
    try:
        # Import torch here since we now know it's available
        import torch
        
        if version == "YOLOv5" and check_package_installed("yolov5"):
            import yolov5
            model = yolov5.load(model_name)
            model_path = MODEL_DIR / model_name
            torch.save(model, model_path)
        elif check_package_installed("ultralytics"):  # YOLOv8 and YOLOv9
            from ultralytics import YOLO
            # Download by constructing the model
            model = YOLO(model_name)
            # Check if download was successful by accessing the model_path
            if hasattr(model, "model") and hasattr(model.model, "pt_path"):
                return True
            elif hasattr(model, "ckpt_path"):
                return True
            else:
                # Manually save to our models directory
                model_path = MODEL_DIR / model_name
                if hasattr(model, "model"):
                    torch.save(model.model, model_path)
                else:
                    torch.save(model, model_path)
        else:
            return False
        
        return True
    except Exception as e:
        print(f"Error downloading model: {e}")
        return False

def get_model_path(version, model_name):
    """Get the full path to a model."""
    # First check if the model exists in the local models directory
    local_path = MODEL_DIR / model_name
    if local_path.exists():
        return str(local_path)
    
    # Check if it might be available through the package
    package = YOLO_VERSIONS[version]["package"]
    
    if package == "yolov5" and check_package_installed("yolov5"):
        try:
            import yolov5
            model_dir = Path(yolov5.__file__).parent / "models"
            package_path = model_dir / model_name
            if package_path.exists():
                return str(package_path)
        except ImportError:
            pass
    
    # For YOLOv8 and YOLOv9, we'll use the model name directly
    # as YOLO() can download it if needed
    return model_name

def get_available_yolo_versions():
    """Get a list of available YOLO versions based on installed packages and models folder."""
    available_versions = set()
    
    # Check installed packages and their models
    if check_package_installed("yolov5"):
        try:
            import yolov5
            model_dir = Path(yolov5.__file__).parent / "models"
            if model_dir.exists() and any(model_dir.glob("*.pt")):
                available_versions.add("YOLOv5")
        except ImportError:
            pass
            
    if check_package_installed("ultralytics"):
        try:
            from ultralytics import YOLO
            # Check if we can access any models
            for version in ["YOLOv8", "YOLOv9"]:
                prefix = YOLO_VERSIONS[version]["model_prefix"]
                # Try to access a small model to verify availability
                try:
                    YOLO(f"{prefix}n.pt")
                    available_versions.add(version)
                except:
                    pass
        except ImportError:
            pass
    
    # Check models folder
    if MODEL_DIR.exists():
        for pt_file in MODEL_DIR.glob("*.pt"):
            if "yolov5" in pt_file.name.lower():
                available_versions.add("YOLOv5")
            elif "yolov8" in pt_file.name.lower():
                available_versions.add("YOLOv8")
            elif "yolov9" in pt_file.name.lower():
                available_versions.add("YOLOv9")
    
    return sorted(list(available_versions))

def load_model(version, model_name):
    """Load a YOLO model from local path or download if needed."""
    try:
        if version == "YOLOv5" and check_package_installed("yolov5"):
            import yolov5
            import torch
            
            # First try to load directly from yolov5 package
            try:
                print(f"Attempting to load {model_name} directly from yolov5 package...")
                model = yolov5.load(model_name)
                # Convert to CPU if needed
                if hasattr(model, 'to'):
                    model = model.to('cpu')
                return model
            except Exception as e:
                print(f"Direct loading failed: {e}")
                
                # Try loading from local path
                local_path = MODEL_DIR / model_name
                if local_path.exists():
                    try:
                        print(f"Loading from local path: {local_path}")
                        model = yolov5.load(str(local_path))
                        if hasattr(model, 'to'):
                            model = model.to('cpu')
                        return model
                    except Exception as e:
                        print(f"Local loading failed: {e}")
                
                # If both attempts failed, try downloading
                try:
                    print(f"Attempting to download {model_name}...")
                    model = yolov5.load(model_name, force_reload=True)
                    # Save the model locally
                    torch.save(model, local_path)
                    if hasattr(model, 'to'):
                        model = model.to('cpu')
                    return model
                except Exception as e:
                    print(f"Download failed: {e}")
                    raise Exception(f"Failed to load or download model: {model_name}")
                    
        elif check_package_installed("ultralytics"):
            from ultralytics import YOLO
            try:
                model = YOLO(model_name)
                if hasattr(model, 'to'):
                    model = model.to('cpu')
                return model
            except Exception as e:
                print(f"Error loading YOLO model: {e}")
                return None
        else:
            raise ImportError(f"Required package for {version} is not installed")
            
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        return None

def get_model_info(model_path):
    """Get information about a model file."""
    try:
        import torch
        checkpoint = torch.load(model_path, map_location='cpu')
        if isinstance(checkpoint, dict):
            # YOLOv5 format
            if 'model' in checkpoint:
                return {
                    'type': 'YOLOv5',
                    'epoch': checkpoint.get('epoch', 'unknown'),
                    'best_fitness': checkpoint.get('best_fitness', 'unknown')
                }
            # YOLOv8 format
            elif 'train_args' in checkpoint:
                return {
                    'type': 'YOLOv8',
                    'epoch': checkpoint.get('epoch', 'unknown'),
                    'best_fitness': checkpoint.get('best_fitness', 'unknown')
                }
        return {'type': 'Unknown', 'epoch': 'unknown', 'best_fitness': 'unknown'}
    except Exception as e:
        print(f"Error reading model info: {e}")
        return {'type': 'Unknown', 'epoch': 'unknown', 'best_fitness': 'unknown'}
