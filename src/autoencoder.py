import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionAutoencoder(nn.Module):
    """
    A simple autoencoder that takes visual latents and reconstructs images.
    This is designed to work with the visual features extracted from the vision transformer.
    """
    
    def __init__(self, latent_dim=4096, image_channels=3, image_size=224, hidden_dims=None):
        super(VisionAutoencoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.image_channels = image_channels
        self.image_size = image_size
        
        if hidden_dims is None:
            hidden_dims = [2048, 1024, 512, 256]
        
        # Decoder network to reconstruct image from latent
        modules = []
        in_channels = latent_dim
        
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim
        
        # Final layer to reconstruct image pixels
        # Assuming the output is flattened image pixels
        self.decoder = nn.Sequential(*modules)
        
        # Calculate the final feature size for reconstruction
        final_feature_size = hidden_dims[-1]
        self.image_flat_size = image_channels * image_size * image_size
        self.reconstruction_layer = nn.Linear(final_feature_size, self.image_flat_size)
        
        # Activation for reconstructed image values
        self.sigmoid = nn.Sigmoid()

    def encode(self, pixel_values):
        """
        In this case, encoding is handled by the main vision model,
        so this just returns the input for compatibility.
        """
        return pixel_values

    def decode(self, latent_features):
        """
        Decode visual latents back to image space.
        Args:
            latent_features: [batch_size, seq_len, feature_dim]
        Returns:
            reconstructed_images: [batch_size, channels, height, width]
        """
        batch_size, seq_len, feature_dim = latent_features.size()
        
        # Reshape to combine batch and sequence dimensions
        latent_flat = latent_features.view(-1, feature_dim)
        
        # Decode through the network
        decoded = self.decoder(latent_flat)
        reconstructed_flat = self.reconstruction_layer(decoded)
        
        # Apply activation to ensure valid image values
        reconstructed_flat = self.sigmoid(reconstructed_flat)
        
        # Reshape back to [batch_size, channels, height, width]
        reconstructed_images = reconstructed_flat.view(
            batch_size, seq_len, self.image_channels, self.image_size, self.image_size
        ).mean(dim=1)  # Average across sequence dimension to get single image
        
        return reconstructed_images

    def forward(self, latent_features):
        reconstructed_images = self.decode(latent_features)
        return reconstructed_images

    def compute_reconstruction_loss(self, original_images, reconstructed_images, loss_type="mse"):
        """
        Compute reconstruction loss between original and reconstructed images.
        Args:
            original_images: [batch_size, channels, height, width]
            reconstructed_images: [batch_size, channels, height, width]
            loss_type: type of loss ("mse", "l1", or "huber")
        Returns:
            loss: scalar reconstruction loss
        """
        if loss_type == "mse":
            loss = F.mse_loss(reconstructed_images, original_images)
        elif loss_type == "l1":
            loss = F.l1_loss(reconstructed_images, original_images)
        elif loss_type == "huber":
            loss = F.huber_loss(reconstructed_images, original_images)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
        return loss


class SimpleVisionDecoder(nn.Module):
    """
    A simpler vision decoder that focuses on reconstructing image patches from visual latents.
    This version works better with the existing vision transformer architecture.
    """
    
    def __init__(self, input_dim=1152, patch_size=14, num_patches=256):
        super(SimpleVisionDecoder, self).__init__()
        
        self.input_dim = input_dim
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.output_dim = patch_size * patch_size * 3  # RGB patches
        
        # Multi-layer decoder to better reconstruct from visual latents
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, self.output_dim),
        )
        
        # Final normalization
        self.norm = nn.LayerNorm(self.output_dim)

    def forward(self, latent_patches):
        """
        Reconstruct image patches from visual latents.
        Args:
            latent_patches: [batch_size, num_patches, input_dim] or [total_patches, input_dim]
        Returns:
            reconstructed_patches: [batch_size, num_patches, patch_size, patch_size, 3] or [total_patches, patch_size, patch_size, 3]
        """
        original_shape = latent_patches.shape
        # Project each patch embedding back to patch pixels
        reconstructed_patches = self.decoder(latent_patches)  # [B*N, output_dim] or [B, N, output_dim]
        
        # Normalize
        reconstructed_patches = self.norm(reconstructed_patches)
        
        # Reshape to [B, N, patch_size, patch_size, 3] or [total_patches, patch_size, patch_size, 3]
        if len(original_shape) == 3:  # [B, N, input_dim]
            B, N, _ = original_shape
            reconstructed_patches = reconstructed_patches.view(B, N, self.patch_size, self.patch_size, 3)
        else:  # [total_patches, input_dim]
            total_patches, _ = original_shape
            reconstructed_patches = reconstructed_patches.view(total_patches, self.patch_size, self.patch_size, 3)
        
        return reconstructed_patches

    def compute_reconstruction_loss(self, original_patches, reconstructed_patches, loss_type="mse"):
        """
        Compute reconstruction loss between original and reconstructed patches.
        Args:
            original_patches: [batch_size, num_patches, patch_size, patch_size, 3] or [total_patches, patch_size, patch_size, 3]
            reconstructed_patches: [batch_size, num_patches, patch_size, patch_size, 3] or [total_patches, patch_size, patch_size, 3]
            loss_type: type of loss ("mse", "l1", or "huber")
        Returns:
            loss: scalar reconstruction loss
        """
        if loss_type == "mse":
            loss = F.mse_loss(reconstructed_patches, original_patches)
        elif loss_type == "l1":
            loss = F.l1_loss(reconstructed_patches, original_patches)
        elif loss_type == "huber":
            loss = F.huber_loss(reconstructed_patches, original_patches)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
        return loss