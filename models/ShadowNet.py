class ShadowNet(nn.Module):
    def __init__(self, num_blocks=7):
        super().__init__()
        # input is 200x200x30
        # 30 is NOT a typo.
        self.netBegin = nn.Sequential(
            nn.Conv2d(30, 64, kernel_size=(3, 3)),
            nn.Relu(),
        )

        self.downBlocks = nn.ModuleList(
            [
                ShadowDownBlock(in_chan=64, out_chan=128),
                ShadowDownBlock(in_chan=128, out_chan=256),
                ShadowDownBlock(in_chan=256, out_chan=512),
                ShadowDownBlock(in_chan=512, out_chan=512),
                ShadowDownBlock(in_chan=512, out_chan=512),
                ShadowDownBlock(in_chan=512, out_chan=512),
                ShadowDownBlock(in_chan=512, out_chan=512),
            ]
        )
        self.upBlocks = nn.ModuleList(
            [
                ShadowUpBlock(in_chan=512, out_chan=512),
                ShadowUpBlock(in_chan=512, out_chan=512),
                ShadowUpBlock(in_chan=512, out_chan=512),
                ShadowUpBlock(in_chan=512, out_chan=512),
                ShadowUpBlock(in_chan=512, out_chan=256),
                ShadowUpBlock(in_chan=256, out_chan=128),
                ShadowUpBlock(in_chan=128, out_chan=64),
            ]
        )

        self.netEnd = nn.Sequential(nn.Conv2d(), nn.Tanh())
        self.residuals = []

    def forward(self, x):
        out = self.netBegin(x)
        self.residuals.append(out)
        for i, d in enumerate(self.downBlocks):
            self.residuals.append(d(self.residuals[i]))

        out = 0
        for i, u in enumerate(self.upBlocks):
            l = len(self.residuals) - 1
            out = u(self.residual[l - i] + out)

        return self.netEnd(self.residuals[0] + out)


class ShadowDownBlock(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()

        self.net = nn.Sequential(
            nn.BatchNorm(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_chan, out_chan, kernel_size=(3, 3)),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class ShadowUpBlock(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size=(3, 3), scale_factor=(2, 2)):
        super().__init__()

        self.net = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor),
            nn.BatchNorm(),
            nn.Conv2d(in_chan, out_chan, kernel_size=kernel_size),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)
