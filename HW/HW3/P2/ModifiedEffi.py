import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s

# Load pre-built EfficientNetV2-S model
original_model = efficientnet_v2_s(pretrained=True)

# Check the original parameter count
original_param_count = sum(p.numel() for p in original_model.parameters())

class ModifiedEfficientNetV2S(nn.Module):
    def __init__(self):
        super(ModifiedEfficientNetV2S, self).__init__()
        # Get the original architecture
        self.model = efficientnet_v2_s(pretrained=False)
        
        # Widen the neural network by multiplying 1.1 to the number output channel of each layers
        self.model.features[2][0].out_channels = int(self.model.features[2][0].out_channels * 1.1)
        self.model.features[3][0].out_channels = int(self.model.features[3][0].out_channels * 1.1)
        self.model.features[4][0].out_channels = int(self.model.features[4][0].out_channels * 1.1)
        self.model.features[5][0].out_channels = int(self.model.features[5][0].out_channels * 1.1)
        
        # Ensure the parameter count remains within 10% of the original
        modified_param_count = sum(p.numel() for p in self.model.parameters())
        assert modified_param_count <= original_param_count * 1.1, \
            f"Parameter count exceeded limit: {modified_param_count} vs {original_param_count * 1.1}"
    
    def forward(self, x):
        return self.model(x)