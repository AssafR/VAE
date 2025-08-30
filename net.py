# Copyright 2019 Stanislav Pidhorskyi
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#  http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ConvBlock(nn.Module):
    """Convolutional block with batch normalization and activation."""
    
    def __init__(self, in_channels: int, out_channels: int, activation: str = "relu"):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.bn(self.conv(x)))


class DeconvBlock(nn.Module):
    """Transposed convolutional block with batch normalization and activation."""
    
    def __init__(self, in_channels: int, out_channels: int, activation: str = "leaky_relu"):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        
        if activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.bn(self.deconv(x)))


class VAE(nn.Module):
    """
    Variational Autoencoder with modern PyTorch architecture.
    
    Args:
        zsize: Dimension of the latent space
        layer_count: Number of convolutional layers in encoder/decoder
        channels: Number of input channels (default: 3 for RGB)
        base_channels: Base number of channels (default: 128)
    """
    
    def __init__(self, zsize: int, layer_count: int = 3, channels: int = 3, base_channels: int = 128):
        super().__init__()
        
        self.zsize = zsize
        self.layer_count = layer_count
        self.channels = channels
        self.base_channels = base_channels
        
        # Calculate channel dimensions for each layer
        self.encoder_channels = [channels]
        
        current_channels = channels
        for i in range(layer_count):
            current_channels = base_channels * (2 ** i)
            self.encoder_channels.append(current_channels)
        
        # Decoder channels should mirror encoder channels (excluding input)
        self.decoder_channels = list(reversed(self.encoder_channels[1:]))
        self.decoder_channels.append(channels)  # Final output layer
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        for i in range(layer_count):
            in_ch = self.encoder_channels[i]
            out_ch = self.encoder_channels[i + 1]
            self.encoder_layers.append(ConvBlock(in_ch, out_ch, activation="relu"))
        
        # Calculate the size after encoding
        self.encoded_size = self.encoder_channels[-1] * 4 * 4
        
        # Latent space projections
        self.fc_mu = nn.Linear(self.encoded_size, zsize)
        self.fc_logvar = nn.Linear(self.encoded_size, zsize)
        
        # Decoder projection
        self.fc_decoder = nn.Linear(zsize, self.encoded_size)
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList()
        for i in range(layer_count - 1):
            in_ch = self.decoder_channels[i]
            out_ch = self.decoder_channels[i + 1]
            self.decoder_layers.append(DeconvBlock(in_ch, out_ch, activation="leaky_relu"))
        
        # Final output layer (no batch norm, tanh activation)
        self.final_layer = nn.ConvTranspose2d(
            self.decoder_channels[-2], 
            channels, 
            kernel_size=4, 
            stride=2, 
            padding=1
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Debug: Print channel dimensions
        print(f"🔧 [ARCHITECTURE] Encoder channels: {self.encoder_channels}")
        print(f"🔧 [ARCHITECTURE] Decoder channels: {self.decoder_channels}")
        print(f"🔧 [ARCHITECTURE] Encoded size: {self.encoded_size}")
    
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights using Xavier/Glorot initialization."""
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input tensor to latent space.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tuple of (mu, logvar) tensors for the latent space
        """
        # Validate input shape
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input tensor, got {x.dim()}D")
        
        # Apply encoder layers
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Flatten and project to latent space
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for sampling from the latent space.
        
        Args:
            mu: Mean tensor
            logvar: Log variance tensor
            
        Returns:
            Sampled latent vector
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to reconstructed image.
        
        Args:
            z: Latent vector of shape (batch_size, zsize)
            
        Returns:
            Reconstructed image tensor
        """
        # Validate input shape
        if z.dim() != 2 or z.size(1) != self.zsize:
            raise ValueError(f"Expected latent tensor of shape (batch_size, {self.zsize}), got {z.shape}")
        
        # Project to spatial dimensions
        x = self.fc_decoder(z)
        x = x.view(x.size(0), self.encoder_channels[-1], 4, 4)
        
        # Apply decoder layers
        for layer in self.decoder_layers:
            x = layer(x)
        
        # Final output layer with tanh activation
        x = self.final_layer(x)
        x = torch.tanh(x)
        
        return x
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tuple of (reconstructed_image, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        
        return reconstructed, mu, logvar
    
    def sample(self, num_samples: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Generate samples from the latent space.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
            
        Returns:
            Generated image tensor
        """
        if device is None:
            device = next(self.parameters()).device
        
        # Sample from standard normal distribution
        z = torch.randn(num_samples, self.zsize, device=device)
        
        # Generate images
        with torch.no_grad():
            samples = self.decode(z)
        
        return samples
    
    def get_latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the latent representation of input images without sampling.
        
        Args:
            x: Input tensor
            
        Returns:
            Mean of the latent representation
        """
        mu, _ = self.encode(x)
        return mu


# Backward compatibility - keep the old function name
def normal_init(m: nn.Module, mean: float, std: float) -> None:
    """Legacy weight initialization function for backward compatibility."""
    if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
        nn.init.normal_(m.weight, mean, std)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
