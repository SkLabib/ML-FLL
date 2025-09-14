import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ResNetModel(nn.Module):
    def __init__(self, num_classes=7, pretrained=True, model_name='resnet18'):
        super(ResNetModel, self).__init__()
        
        # Load pretrained ResNet model based on model_name
        if model_name == 'resnet18':
            self.resnet = models.resnet18(pretrained=pretrained)
        elif model_name == 'resnet50':
            self.resnet = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported ResNet model: {model_name}")
        
        # Replace the final fully connected layer
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)
    
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