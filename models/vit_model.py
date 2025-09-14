import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ViTModel(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        super(ViTModel, self).__init__()
        
        # Load pretrained ViT model
        self.vit = models.vit_b_16(pretrained=pretrained)
        
        # Replace the final classification head
        in_features = self.vit.heads.head.in_features
        self.vit.heads.head = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.vit(x)
    
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