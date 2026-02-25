# model_architecture.py

# imports
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class TemporalUnetTransformer(nn.Module):
    """
    Spatiotemporal U-Net with a temporal Transformer at the bottleneck.

    This model combines a convolutional U-Net encoder–decoder architecture
    with a Transformer-based temporal module applied at the deepest spatial
    representation. The encoder is applied independently to each time step,
    while temporal dependencies are learned at the bottleneck level using
    self-attention.

    The output is a spatial field corresponding to the prediction at the
    next time step.
    """

    def __init__(
                 self,
                 in_channels=1,
                 out_channels=1,
                 encoder_name="resnet18",
                 encoder_weights=None,
                 n_heads=4,
                 num_layers=1,
                 ):
        """
        Initialize the Temporal U-Net Transformer model.

        Parameters
        ----------
        in_channels : int, optional
            Number of input channels per time step. Default is 1.
        out_channels : int, optional
            Number of output channels. Default is 1.
        encoder_name : str, optional
            Name of the encoder backbone used in the U-Net architecture.
            Must be compatible with `segmentation_models_pytorch`.
            Default is "resnet34".
        encoder_weights : str or None, optional
            Pretrained weights for the encoder backbone.
            Common options are "imagenet" or None. Default is None.
        n_heads : int, optional
            Number of attention heads in the temporal Transformer.
            Default is 8.
        num_layers : int, optional
            Number of Transformer encoder layers.
            Default is 2.
        """

        super().__init__()

        # U-Net backbone
        self.unet = smp.Unet(
                             encoder_name=encoder_name,
                             encoder_weights=encoder_weights,
                             in_channels=in_channels,
                             classes=out_channels,
                             activation=None,
                             )

        self.encoder = self.unet.encoder
        self.decoder = self.unet.decoder
        self.head = self.unet.segmentation_head

        # Number of channels at the bottleneck (deepest encoder level)
        bottleneck_channels = self.encoder.out_channels[-1]

        # Temporal Transformer
        encoder_layer = nn.TransformerEncoderLayer(
                                                   d_model=bottleneck_channels,
                                                   nhead=n_heads,
                                                   dim_feedforward=4 * bottleneck_channels,
                                                   batch_first=True,
                                                   dropout=0.1,
                                                   )

        self.temporal_transformer = nn.TransformerEncoder(
                                                          encoder_layer,
                                                          num_layers=num_layers,
                                                          )

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (B, T, C, H, W), where:
            - B is the batch size
            - T is the number of time steps
            - C is the number of input channels
            - H is the height of the spatial grid
            - W is the width of the spatial grid

        Returns
        -------
        torch.Tensor
            Output tensor with shape (B, out_channels, H, W),
            representing the predicted spatial field at the next time step.
        """

        B, T, _, H, W = x.shape

        bottlenecks = []
        skips_all = []

        # Encoder applied per time step
        for t in range(T):
            feats = self.encoder(x[:, t])   # multi-scale feature maps
            skips_all.append(feats[:-1])    # skip connections
            bottlenecks.append(feats[-1])   # bottleneck features (B, C, h, w)

        # Stack bottleneck features along the temporal dimension
        # Shape: (B, T, C, h, w)
        Z = torch.stack(bottlenecks, dim=1)

        B, T, C, h, w = Z.shape

        # Temporal Transformer
        # Each spatial location is treated as a temporal sequence
        Z = Z.permute(0, 3, 4, 1, 2)      # (B, h, w, T, C)
        Z = Z.reshape(B * h * w, T, C)    # (B*h*w, T, C)

        Z = self.temporal_transformer(Z)  # (B*h*w, T, C)
        Z = Z[:, -1]                      # last temporal step

        # Restore spatial structure
        Z = Z.reshape(B, h, w, C)
        Z = Z.permute(0, 3, 1, 2)         # (B, C, h, w)

        # U-Net decoder
        # Uses skip connections from the last time step
        features = skips_all[-1] + [Z]

        y = self.decoder(features)
        y = self.head(y)

        return y
