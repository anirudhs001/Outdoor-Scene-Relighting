# models for image relighting pipeline

# simpler blocks
# conv2d block
class Conv2dBlock(nn.Module):
    """
    in_dim: number of neurons
    out_dim: Kernel size
    stride: Strid
    kernel_size
    padding: (bool) = True for adding padding
    norm: Normalization to use(BatchNorm/Instance Norm etc)
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size,
        stride,
        padding,
        norm=None,
        leaky_relu_neg_slope=0.2,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            padding_mode="reflect",
        )
        self.norm = None
        if norm == "batch":
            self.norm = nn.BatchNorm2d(out_dim)
        if norm == "instance":
            self.norm = nn.InstanceNorm2d(out_dim)
        self.activn = nn.LeakyReLU(leaky_relu_neg_slope)

    def forward(self, x):
        # print(x.shape)
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.activn(x)
        return x


# resblock
class ResBlock(nn.Module):
    def __init__(self, dim, kernel_size, stride, padding):
        super().__init__()

        # conv with batchnorm and leaky relu activation
        self.conv1 = Conv2dBlock(
            dim, dim, kernel_size=kernel_size, stride=stride, padding=padding
        )
        # simple convolution and batchnorm
        self.conv2 = nn.Conv2d(
            dim, dim, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.norm(out)
        # print(out.shape)
        # print(x.shape)
        out += x
        return out
