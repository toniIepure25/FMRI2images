"""
Simple 3D CNN for fMRI to CLIP Embedding Mapping.

This is a baseline CNN architecture that processes 3D fMRI volumes
and outputs CLIP embeddings.

Architecture:
    3D Conv layers → Flatten → FC layers → L2-normalize → CLIP embedding (512-dim)

Author: Bachelor Thesis - fMRI to Image Reconstruction
"""

import torch
import torch.nn as nn
from typing import Tuple


class SimpleCNN(nn.Module):
    """
    Simple 3D CNN for fMRI → CLIP embedding mapping.
    
    This baseline model uses 3D convolutions to process fMRI brain volumes
    and produces 512-dimensional CLIP embeddings.
    
    Args:
        input_shape: Shape of input fMRI volume (channels, height, width, depth)
                    e.g., (1, 81, 104, 83) for NSD data
        output_dim: Output dimension (default: 512 for CLIP ViT-B/32)
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int, int] = (1, 81, 104, 83),
        output_dim: int = 512
    ):
        super().__init__()
        
        self.input_shape = input_shape
        self.output_dim = output_dim
        
        # 3D Convolutional layers
        self.conv_layers = nn.Sequential(
            # Conv block 1: (1, 81, 104, 83) -> (32, 40, 51, 41)
            nn.Conv3d(input_shape[0], 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # Conv block 2: (32, 40, 51, 41) -> (64, 20, 25, 20)
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # Conv block 3: (64, 20, 25, 20) -> (128, 10, 12, 10)
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
            
            # Conv block 4: (128, 10, 12, 10) -> (256, 5, 6, 5)
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        
        # Calculate flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            conv_output = self.conv_layers(dummy_input)
            self.flattened_size = conv_output.view(1, -1).shape[1]
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, output_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input fMRI volume [batch, channels, height, width, depth]
        
        Returns:
            CLIP embedding [batch, output_dim]
        """
        # Convolutional feature extraction
        x = self.conv_layers(x)
        
        # Fully connected layers
        x = self.fc_layers(x)
        
        # L2 normalization (important for CLIP space alignment)
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        
        return x
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_model():
    """Test the model with dummy data."""
    print("Testing SimpleCNN model...")
    
    # Create model
    model = SimpleCNN(input_shape=(1, 81, 104, 83), output_dim=512)
    print(f"✓ Model created")
    print(f"  Total parameters: {model.get_num_parameters():,}")
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 1, 81, 104, 83)
    output = model(dummy_input)
    
    print(f"✓ Forward pass successful")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output L2 norm: {torch.norm(output, p=2, dim=1).mean():.4f} (should be ~1.0)")
    
    assert output.shape == (batch_size, 512), f"Wrong output shape: {output.shape}"
    assert torch.allclose(torch.norm(output, p=2, dim=1), torch.ones(batch_size), atol=1e-5), "Output not normalized"
    
    print("✅ All tests passed!")


if __name__ == "__main__":
    test_model()
