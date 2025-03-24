from ultralytics import YOLO
from PIL import Image
import os
import cv2
import numpy as np
import torch
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.conv import Conv, Concat
from ultralytics.nn.modules.block import C2f, C3, SPPF, Bottleneck
from ultralytics.nn.modules.head import Detect
from torch.nn import (
    Sequential, Conv2d, BatchNorm2d, SiLU, Module,
    ModuleList, ReLU, Upsample, MaxPool2d
)

# Store the original torch.load function
_original_torch_load = torch.load

# Override torch.load to use weights_only=False
def custom_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = custom_torch_load

# Add required classes to safe globals for PyTorch 2.6
safe_classes = [
    DetectionModel, Sequential, Conv2d, BatchNorm2d,
    SiLU, Module, ModuleList, ReLU, Upsample, MaxPool2d,
    Conv, getattr, C2f, C3, SPPF, Bottleneck, Concat, Detect
]
torch.serialization.add_safe_globals(safe_classes)

class ModelHandler:
    def __init__(self, model_path='best.pt'):
        self.model = YOLO(model_path)
        # Print model information
        print("Model loaded successfully")
        print(f"Model classes: {self.model.names}")
        
        # Use the model's class names instead of hardcoded ones
        self.class_names = list(self.model.names.values())
        self.last_bbox = None
    
    def predict(self, image_path):
        """Run prediction on an image"""
        try:
            # Run prediction
            results = self.model.predict(source=image_path, conf=0.25)
            
            # Process results
            for r in results:
                for box in r.boxes:
                    class_id = int(box.cls)
                    confidence = float(box.conf)
                    class_name = self.model.names[class_id]
                    
                    # Format the result with separate confidence
                    return f"Road sign detected: {class_name} (Confidence: {confidence:.2f})"
            
            return "No road signs detected"
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return "Error during prediction"

    def get_last_bbox(self):
        return self.last_bbox 