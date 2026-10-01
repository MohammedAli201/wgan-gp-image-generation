import torch.nn as nn


class CriticNetwork(nn.Module):
    """
    The CriticNetwork class is a convolutional neural network (CNN) used as a critic or discriminator in Generative Adversarial Networks (GANs).
    It aims to differentiate between real and generated (fake) images. The network architecture involves several convolutional layers
    that progressively downsample the input image.

    Parameters:
    - input_channels: The number of channels in the input images (e.g., 3 for RGB images).
    - base_feature_count: The base number of features (filters) for the convolutional layers. Subsequent layers
      use multiples of this base count.

    The network employs LeakyReLU for non-linear activation and InstanceNorm2d for normalizing the outputs of convolution layers.
    It consists of a series of convolutional blocks (Conv2d + InstanceNorm2d + LeakyReLU), and the final Conv2d layer
    outputs a single value representing the perceived authenticity of the input image.
    """

    def __init__(self, input_channels, base_feature_count):
        super(CriticNetwork, self).__init__()
        self.disc_layers = nn.Sequential(
            nn.Conv2d(input_channels, base_feature_count,
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            self._create_disc_block(
                base_feature_count, base_feature_count * 2, 4, 2, 1),
            self._create_disc_block(
                base_feature_count * 2, base_feature_count * 4, 4, 2, 1),
            self._create_disc_block(
                base_feature_count * 4, base_feature_count * 8, 4, 2, 1),
            nn.Conv2d(base_feature_count * 8, 1,
                      kernel_size=4, stride=2, padding=0)
        )

    def _create_disc_block(self, input_channels, output_channles, kernel_size, stride, padding):
        """
        Creates a convolutional block for the discriminator network. Each block consists of a Conv2d layer,
        an InstanceNorm2d layer, and a LeakyReLU activation layer.

        Parameters:
        - in_channels: Number of input channels to the convolutional layer.
        - out_channels: Number of output channels from the convolutional layer.
        - kernel_size, stride, padding: Convolutional layer parameters.
        """

        """
         Leaky ReLU allows a small, non-zero gradient when the unit is not active. 
         This property is useful for the discriminator as it helps in maintaining gradient 
         flow during backpropagation, which can prevent the dying 
         ReLU problem (where neurons stop learning completely) 
         and helps in making the discriminator more robust.
        """
        return nn.Sequential(
            nn.Conv2d(input_channels, output_channles,
                      kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(output_channles, affine=True),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        """
        Forward pass of the CriticNetwork. Processes an input image tensor through the discriminator layers to produce
        a scalar value representing the "realness" of the image.

        Parameters:
        - x: Input image tensor.
        """
        output = self.disc_layers(x)
        return output
