import torch.nn as nn





class DASCA(nn.Module):
    def __init__(self, inp, kernel_size=1, ratio=1, band_kernel_size=11, dw_size=(1, 1), padding=(0, 0), stride=1,
                 square_kernel_size=3, relu=True):
        super(DASCA, self).__init__()
        self.dwconv1 = nn.Conv2d(inp, inp, square_kernel_size, padding=square_kernel_size // 2, groups=inp)
        self.dwconv_h = nn.Conv2d(in_channels=inp, out_channels=inp, kernel_size=(1, 3), padding=(0, 1))
        self.dwconv_w = nn.Conv2d(in_channels=inp, out_channels=inp, kernel_size=(3, 1), padding=(1, 0))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.conv_h = nn.Conv2d(in_channels=inp, out_channels=inp, kernel_size=(1, 3), padding=(0, 1))
        self.conv_w = nn.Conv2d(in_channels=inp, out_channels=inp, kernel_size=(3, 1), padding=(1, 0))

        gc = inp // ratio
        self.excite = nn.Sequential(
            nn.Conv2d(inp, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2), groups=gc),
            nn.BatchNorm2d(gc),
            nn.ReLU(inplace=True),
            nn.Conv2d(gc, inp, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0), groups=gc),
            # nn.Sigmoid()
        )

        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def sge(self, x):
        # [N, D, C, 1]
        # print("0", x.shape)
        conv_h = self.conv_h(x)
        # print("111", conv_h.shape)
        x_h = self.pool_h(conv_h)

        conv_w = self.conv_w(x)
        x_w = self.pool_w(conv_w)

        x_gather = x_h + x_w  # .repeat(1,1,1,x_w.shape[-1])
        x_gather = self.dwconv1(x_gather)

        conv_h = self.dwconv_h(conv_h)
        conv_h = self.sigmoid_h(conv_h)

        conv_w = self.dwconv_w(conv_w)
        conv_w = self.sigmoid_w(conv_w)

        out1 = conv_h * x_gather
        out2 = conv_w * x_gather
        out = out1 + out2

        ge = self.excite(out) + x  # [N, 1, C, 1]

        return ge

    def forward(self, x):
        # print("333", x.shape)
        # loc = self.dwconv(x)
        # print("444", loc.shape)
        att = self.sge(x)
        # print("555", att.shape)
        out = att + x

        return out