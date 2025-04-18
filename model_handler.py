import torch
from ultralytics import YOLO
import os

class ModelHandler:
    def __init__(self):
        # Initialize YOLO model
        self.model = YOLO('best.pt')
        
        # Set device (GPU if available, else CPU)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        # Enable model optimization
        if hasattr(self.model, 'fuse'):
            self.model.fuse()  # Fuse Conv2d + BatchNorm2d layers for faster inference
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Enable model optimization if available
        if hasattr(self.model, 'optimize'):
            self.model.optimize()
    
    def predict(self, image_path):
        try:
            # Run inference with optimized settings
            results = self.model.predict(
                source=image_path,
                conf=0.25,
                iou=0.45,
                verbose=False,
                device=self.device,
                half=True,
                agnostic_nms=True,
                max_det=50,
                stream=True
            )
            
            # Process the first result (since we're only processing one image)
            for r in results:
                boxes = r.boxes
                if len(boxes) == 0:
                    return {
                        'message': 'No objects detected',
                        'bbox': [],
                        'class_name': [],
                        'confidence': []
                    }
                
                # Get the first detection (highest confidence)
                box = boxes[0]
                bbox = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = self.model.names[cls]
                
                return {
                    'message': f'Detected {class_name} with confidence {conf:.2f}',
                    'bbox': bbox,
                    'class_name': class_name,
                    'confidence': conf
                }
            
            return {
                'message': 'No results found',
                'bbox': [],
                'class_name': [],
                'confidence': []
            }
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            return {
                'message': f'Error during prediction: {str(e)}',
                'bbox': [],
                'class_name': [],
                'confidence': []
            }