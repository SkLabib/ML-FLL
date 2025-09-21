from .cnn_model import SkinLesionCNN
from .resnet_model import ResNetModel
from .vit_model import ViTModel
from .efficientnet_model import EfficientNetModel

def create_model(model_type='efficientnet', model_name=None, num_classes=7, pretrained=True, grayscale=False, 
                 use_fedbn=True, num_clusters=4, client_id=0):
    """
    Factory function to create a model based on the specified type.
    
    Args:
        model_type (str): Type of model to create ('cnn', 'resnet', 'resnet18', 'resnet50', 'mobilenet', 'efficientnet')
        model_name (str): Specific model variant (e.g., 'efficientnet-b0', 'efficientnet-b1', 'mobilenet_v2')
        num_classes (int): Number of output classes (default: 7 to support all datasets)
        pretrained (bool): Whether to use pretrained weights for transfer learning
        grayscale (bool): Whether to convert input to grayscale
        use_fedbn (bool): Whether to use FedBN (local BatchNorm layers)
        num_clusters (int): Number of clusters for cluster-specific heads
        client_id (int): Client ID for initialization
        
    Returns:
        model: The created model instance
    """
    # Validate model_type to prevent unauthorized access
    allowed_types = ['cnn', 'resnet', 'resnet18', 'resnet50', 'mobilenet', 'efficientnet']
    if model_type not in allowed_types:
        raise ValueError(f"Unauthorized model type: {model_type}. Allowed types: {allowed_types}")
    if model_type == 'cnn':
        return SkinLesionCNN(num_classes=num_classes)
    elif model_type == 'resnet18' or model_type == 'resnet':
        model_name = model_name if model_name else 'resnet18'
        return ResNetModel(num_classes=num_classes, pretrained=pretrained, model_name=model_name)
    elif model_type == 'resnet50':
        model_name = model_name if model_name else 'resnet50'
        return ResNetModel(num_classes=num_classes, pretrained=pretrained, model_name=model_name)
    elif model_type == 'mobilenet':
        # Use MobileNetV2 from EfficientNet implementation with FedBN and cluster support
        model_name = model_name if model_name else 'mobilenet_v2'
        return EfficientNetModel(
            num_classes=num_classes, 
            pretrained=pretrained, 
            model_name=model_name, 
            grayscale=grayscale,
            use_fedbn=use_fedbn,
            num_clusters=num_clusters,
            client_id=client_id
        )
    elif model_type == 'efficientnet':
        model_name = model_name if model_name else 'efficientnet-b0'
        return EfficientNetModel(
            num_classes=num_classes, 
            pretrained=pretrained, 
            model_name=model_name, 
            grayscale=grayscale,
            use_fedbn=use_fedbn,
            num_clusters=num_clusters,
            client_id=client_id
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")