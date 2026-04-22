import torch
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
import numpy as np

class FeatureExtractor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1])) 
        self.model.to(self.device)
        self.model.eval()
        
        self.transforms = T.Compose([
            T.ToPILImage(),
            T.Resize((128, 64)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract(self, frame, bbox):
        """Crops the bounding box from the frame and returns a normalized 512-d feature vector."""
        x1, y1, x2, y2 = map(int, bbox)
        
       
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        crop = frame[y1:y2, x1:x2]
        
        
        if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
            return np.zeros(512)

        
        crop_t = self.transforms(crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(crop_t).squeeze().cpu().numpy()
            
        
        norm = np.linalg.norm(feat)
        if norm > 0:
            return feat / norm
        return feat