
import torch
import torch.nn as nn
# Generator


import torch.nn as nn


class Generator(nn.Module):
    """
    Generator class for a Generative Adversarial Network (GAN).

    The generator aims to produce images from a latent space representation. It typically
    involves upscaling from a lower dimensional space to the high-dimensional space of
    image data. This is achieved through a series of transposed convolutional layers that
    gradually upscale the input to the desired output image size.

    Parameters:
    - z_dim: Dimension of the latent space representation. - A latent vector is  noise vector or latent space representation, is a randomly generated array of numbers. It serves as the input to the generator.
        This vector is typically sampled from a standard normal distribution (Gaussian distribution)
    - img_channels: Number of channels in the output images (e.g., 3 for RGB images).
    - feature_base_count: Base number of features which is used to calculate the number of
      output channels for each transposed convolutional layer in the generator.


    """

    def __init__(self, z_dim, img_channels, feature_base_count):
        super(Generator, self).__init__()
        # Building the generator network using a series of 'blocks' (transposed convolutions)
        self.gen = nn.Sequential(
            self._transposed_conv_block(
                z_dim, feature_base_count * 16, 4, 1, 0),
            self._transposed_conv_block(
                feature_base_count * 16, feature_base_count * 8, 4, 2, 1),
            self._transposed_conv_block(
                feature_base_count * 8, feature_base_count * 4, 4, 2, 1),
            self._transposed_conv_block(
                feature_base_count * 4, feature_base_count * 2, 4, 2, 1),
            nn.ConvTranspose2d(feature_base_count * 2, img_channels, 4, 2, 1),
            # Tanh activation to output pixel values in range [-1, 1]
            nn.Tanh()
        )

    def _transposed_conv_block(self, trans_in_channels, trans_out_channels, kernel_sz, stride_sz, padding_sz):
        """
        A helper function to create a transposed convolutional block. Each block consists of
        a transposed convolutional layer, a batch normalization layer, and a ReLU activation.

        Parameters:
        - trans_in_channels: Number of input channels to the transposed convolutional layer.
        - trans_out_channels: Number of output channels from the transposed convolutional layer.
        - kernel_sz: Size of the kernel (filter) for the transposed convolution.
        - stride_sz: Stride size for the transposed convolution.
        - padding_sz: Padding size for the transposed convolution.
        """
        return nn.Sequential(
            nn.ConvTranspose2d(trans_in_channels, trans_out_channels,
                               kernel_sz, stride_sz, padding_sz, bias=False),
            # Normalizes output for stable training
            nn.BatchNorm2d(trans_out_channels),
            nn.ReLU()  # ReLU activation to introduce non-linearity
        )

    def forward(self, x):
        """
        Forward pass of the generator. Takes in a latent vector and produces an image.

        Parameters:
        - x: Latent space representation (tensor).
        """
        return self.gen(x)


"""

code  checks if the current m is an instance fo nn.ConvTranspoe2d or nn.BatchNorm2d
  which is used for upsampling and normalizing the data respectively.
  if the condition is met then the transposed convolutional layer is initialized with a normal distribution its weight are initalzed using a normal distribution with mean of 0.0 and standard deviation of 0.02

"""
