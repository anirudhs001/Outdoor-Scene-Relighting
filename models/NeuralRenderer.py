from models.basic_nets import Conv2dBlock


class NeuralRenderer(nn.Module):
    def __init__(self, num_blocks=4):
        super().__init__()
        # input is 200x200x14
        self.netBegin = nn.Sequential(
            Conv2dBlock(in_dim=14, out_dim=32, kernel_size=(3, 3), norm="group"),
            Conv2dBlock(in_dim=32, out_dim=32, kernel_size=(3, 3), norm="group"),
            Conv2dBlock(in_dim=32, out_dim=32, kernel_size=(3, 3), norm="group"),
        )

        self.downBlocks = nn.ModuleList(
            [
                NRDownBlock(in_chan=32, out_chan=64),
                NRDownBlock(in_chan=64, out_chan=128),
                NRDownBlock(in_chan=128, out_chan=256),
                NRDownBlock(in_chan=256, out_chan=512),
            ]
        )
        self.upBlocks = nn.ModuleList(
            [
                NRUpBlock(in_chan=512, out_chan=256),
                NRUpBlock(in_chan=256, out_chan=128),
                NRUpBlock(in_chan=128, out_chan=64),
                NRUpBlock(in_chan=64, out_chan=32),
            ]
        )

        self.netEnd = nn.Sequential(
            Conv2dBlock(in_dim=32, out_dim=32),
            Conv2dBlock(in_dim=32, out_dim=32),
            Conv2dBlock(in_dim=32, out_dim=3),
        )

        self.residuals = []

    def forward(self, x):
        out = self.netBegin(x)
        self.residuals.append(out)
        for i, d in enumerate(self.downBlocks):
            self.residuals.append(d(self.residuals[i]))

        out = 0
        for i, u in enumerate(self.upBlocks):
            out = u(self.residual[-1 - i] + out)

        return self.netEnd(out)


class NRDownBlock(nn.Module):
    def __init__(
        self, in_chan, out_chan, kernel_size=(3, 3), pool_kernel=(2, 2), norm="group"
    ):
        super().__init__()

        self.net = nn.Sequential(
            Conv2dBlock(
                in_dim=in_chan, out_dim=in_dim, kernel_size=kernel_size, norm=norm
            ),
            nn.Maxpool2d(kernel_size=pool_kernel),
            Conv2dBlock(
                in_dim=in_chan, out_dim=out_chan, kernel_size=kernel_size, norm=norm
            ),
            Conv2dBlock(
                in_dim=out_chan, out_dim=out_chan, kernel_size=kernel_size, norm=norm
            ),
        )

    def forward(self, x):
        return self.net(x)


class NRUpBlock(nn.Module):
    def __init__(
        self, in_chan, out_chan, kernel_size=(3, 3), scale_factor=(2, 2), norm="group"
    ):
        super().__init__()

        self.net = nn.Sequential(
            Conv2dBlock(
                in_dim=in_chan, out_dim=in_chan, kernel_size=kernel_size, norm=norm
            ),
            Conv2dBlock(
                in_dim=in_chan, out_dim=out_chan, kernel_size=kernel_size, norm=norm
            ),
            nn.Upsample(scale_factor=scale_factor),
            Conv2dBlock(
                in_dim=out_chan, out_dim=out_chan, kernel_size=kernel_size, norm=norm
            ),
        )

    def forward(self, x):
        return self.net(x)
