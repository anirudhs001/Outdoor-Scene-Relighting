from .basic_nets import Conv2dBlock, ResBlock


class InverseRenderingNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.net_base = nn.Sequential(
            Conv2dBlock(in_dim=3, out_dim=64, kernel_size=(7, 7), norm="instance"),
            Conv2dBlock(
                in_dim=64, out_dim=128, kernel_size=(3, 3), stride=2, norm="instance"
            ),
            Conv2dBlock(
                in_dim=128, out_dim=256, kernel_size=(3, 3), stride=2, norm="instance"
            ),
            ResBlock(dim=256, kernel_size=(3, 3), norm="instance"),
            ResBlock(dim=256, kernel_size=(3, 3), norm="instance"),
            ResBlock(dim=256, kernel_size=(3, 3), norm="instance"),
            ResBlock(dim=256, kernel_size=(3, 3), norm="instance"),
            ResBlock(dim=256, kernel_size=(3, 3), norm="instance"),
            ResBlock(dim=256, kernel_size=(3, 3), norm="instance"),
            ResBlock(dim=256, kernel_size=(3, 3), norm="instance"),
            ResBlock(dim=256, kernel_size=(3, 3), norm="instance"),
            ResBlock(dim=256, kernel_size=(3, 3), norm="instance"),
        )
        self.net_albedo = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv2dBlock(in_dim=256, out_dim=128, kernel_size=(3, 3)),
            nn.Upsample(scale_factor=2),
            Conv2dBlock(in_dim=128, out_dim=64, kernel_size=(3, 3)),
            nn.Conv2d(64, 3, kernel_size=(7, 7)),
        )
        self.net_normal = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv2dBlock(in_dim=256, out_dim=128, kernel_size=(3, 3)),
            nn.Upsample(scale_factor=2),
            Conv2dBlock(in_dim=128, out_dim=64, kernel_size=(3, 3)),
            nn.Conv2d(64, 3, kernel_size=(7, 7)),
        )
        self.net_shadow = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv2dBlock(in_dim=256, out_dim=128, kernel_size=(3, 3)),
            nn.Upsample(scale_factor=2),
            Conv2dBlock(in_dim=128, out_dim=64, kernel_size=(3, 3)),
            nn.Conv2d(64, 3, kernel_size=(7, 7)),
        )

        def forward(self, x):
            out = self.net_base(x)
            albedo = self.net_albedo(out)
            normal = self.net_shadow(out)
            shadow = self.net_shadow(out)

            return (albedo, normal, shadow)
