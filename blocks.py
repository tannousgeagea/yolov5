import torch
import torch.nn as nn

class ConvBNSiLU(nn.Module):
    """
    This class implements a convolutional block that includes a Convolutional layer followed by Batch Normalization and SiLU activation.

    Parameters:
    in_channels (int): Number of channels in the input image
    out_channels (int): Number of channels produced by the convolution
    kernel_size (int): Size of the convolving kernel
    stride (int): Stride of the convolution
    padding (int): Zero-padding added to both sides of the input
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBNSiLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()  # SiLU / Swish activation

    def forward(self, x):
        """
        Forward pass of the ConvBNSiLU block.

        Parameters:
        x (Tensor): The input tensor

        Returns:
        Tensor: The output tensor after applying convolution, batch normalization, and SiLU activation
        """
        return self.silu(self.bn(self.conv(x)))

class Focus(nn.Module):
    """
    This class implements the Focus layer, which slices the input tensor into four smaller tensors along the height and width dimensions,
    concatenates these tensors along the channel dimension, and then applies a convolutional operation.

    Parameters:
    in_channels (int): Number of channels in the input image
    out_channels (int): Number of channels produced by the convolution after focusing
    kernel_size (int): Size of the convolving kernel
    stride (int): Stride of the convolution
    padding (int): Zero-padding added to both sides of the input for the convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Focus, self).__init__()
        # The number of input channels is multiplied by 4 because the input is sliced into 4 parts
        self.conv = ConvBNSiLU(in_channels * 4, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        """
        Forward pass of the Focus layer.

        Parameters:
        x (Tensor): The input tensor

        Returns:
        Tensor: The output tensor after focusing and applying a convolutional block
        """
        # Split the input tensor into four smaller tensors and concatenate them along the channel dimension
        patch_top_left = x[..., ::2, ::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right), dim=1)
        return self.conv(x)

class Bottleneck(nn.Module):
    """
    This class represents a single bottleneck block with a residual connection.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        shortcut (bool): If True, use the residual shortcut.
        e (float): Expansion factor for the bottleneck.
    """
    def __init__(self, in_channels, out_channels, shortcut=True, e=0.5):
        super(Bottleneck, self).__init__()
        hidden_channels = int(out_channels * e)
        self.conv1 = ConvBNSiLU(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBNSiLU(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        """
        Forward pass of the Bottleneck block.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after passing through the bottleneck block.
        """
        y = self.conv1(x)
        y = self.conv2(y)
        if self.use_add:
            y += x
        return y

class C3(nn.Module):
    """
    This class represents the C3 module, a CSP bottleneck with three convolutional blocks.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        n (int): Number of bottleneck blocks.
        shortcut (bool): If True, use the residual shortcut in bottleneck blocks.
        e (float): Expansion factor for the bottleneck blocks.
    """
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, e=0.5):
        super(C3, self).__init__()
        self.conv1 = ConvBNSiLU(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBNSiLU(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = ConvBNSiLU(2 * out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bottlenecks = nn.Sequential(
            *[Bottleneck(out_channels, out_channels, shortcut, e) for _ in range(n)]
        )

    def forward(self, x):
        """
        Forward pass of the C3 module.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after passing through the C3 module.
        """
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x1 = self.bottlenecks(x1)
        x = torch.cat((x1, x2), dim=1)  # Merge the two paths
        return self.conv3(x)

class BottleneckCSP(nn.Module):
    """
    CSP Bottleneck that aggregates the features from a standard bottleneck and a shortcut connection.

    Parameters:
    in_channels (int): Number of channels in the input
    out_channels (int): Number of channels in the output
    n (int): Number of bottleneck blocks to use
    shortcut (bool): If True, use the residual shortcut in bottleneck blocks.
    e (float): Expansion factor for the bottleneck blocks.
    """

    def __init__(self, in_channels, out_channels, n=1, shortcut=True, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(out_channels * e)  # hidden channels
        self.cv1 = ConvBNSiLU(in_channels, c_, 1, 1)
        self.cv2 = nn.Conv2d(in_channels, c_, 1, 1, bias=False)
        self.cv3 = ConvBNSiLU(in_channels, c_, 1, 1, bias=False)
        self.cv4 = ConvBNSiLU(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))

class SPPF(nn.Module):
    """
    This class implements the SPPF (Spatial Pyramid Pooling - Fast) block, which applies max pooling at different kernel sizes.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_sizes (tuple of int): Tuple of kernel sizes for max pooling.
    """
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13)):
        super(SPPF, self).__init__()
        c_ = in_channels // 2
        self.conv1 = ConvBNSiLU(in_channels, c_, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBNSiLU(c_ * 4, out_channels, kernel_size=1, stride=1, padding=0)
        self.maxpool_layers = nn.ModuleList([nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes])

    def forward(self, x):
        """
        Forward pass of the SPPF block.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after passing through the SPPF block.
        """
        x = self.conv1(x)
        x = torch.cat([x] + [maxpool(x) for maxpool in self.maxpool_layers], dim=1)
        return self.conv2(x)

class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)