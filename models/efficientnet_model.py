import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class EfficientNetModel(nn.Module):
    def __init__(self, num_classes=7, pretrained=True, model_name='efficientnet-b0', grayscale=False):
        super(EfficientNetModel, self).__init__()
        
        # Always add the same conversion layer architecture for compatibility
        # All models will expect 3-channel input (RGB)
        # Grayscale images will be converted to 3-channel in data preprocessing
        self.grayscale = grayscale
        # Remove the conversion layer entirely - handle channel conversion in data preprocessing
        # This ensures all models have identical architectures
        
        # Load pretrained model based on model_name
        if model_name == 'efficientnet-b0':
            self.efficientnet = models.efficientnet_b0(pretrained=pretrained)
        elif model_name == 'efficientnet-b1':
            self.efficientnet = models.efficientnet_b1(pretrained=pretrained)
        elif model_name == 'efficientnet-b2':
            self.efficientnet = models.efficientnet_b2(pretrained=pretrained)
        elif model_name == 'mobilenet_v2':
            self.efficientnet = models.mobilenet_v2(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Replace the final classifier
        if 'efficientnet' in model_name:
            in_features = self.efficientnet.classifier[1].in_features
            self.efficientnet.classifier = nn.Sequential(
                nn.Dropout(p=0.5, inplace=True),
                nn.Linear(in_features, num_classes)
            )
        elif model_name == 'mobilenet_v2':
            in_features = self.efficientnet.classifier[1].in_features
            self.efficientnet.classifier = nn.Sequential(
                nn.Dropout(p=0.5, inplace=True),
                nn.Linear(in_features, num_classes)
            )
    
    def forward(self, x):
        # All inputs should now be 3-channel (RGB)
        # Grayscale conversion is handled in data preprocessing
        return self.efficientnet(x)
    
    def get_weights(self):
        """
        Get model weights as a list of tensors.
        """
        return [param.data.clone() for param in self.parameters()]
    
    def set_weights(self, weights):
        """
        Set model weights from a list of tensors.
        """
        for param, weight in zip(self.parameters(), weights):
            param.data = weight.to(param.device).clone()
