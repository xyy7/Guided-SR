# -*- coding: utf-8 -*-
"""
@Author  :   zhwzhong
@License :   (C) Copyright 2013-2018, hit
@Contact :   zhwzhong@hit.edu.cn
@Software:   PyCharm
@File    :   nafnet.py
@Time    :   2023/2/1 20:08
@Desc    :
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .common import ConvBNReLU2D
    from .local_arch import Local_Base
except:
    from common import ConvBNReLU2D
    from local_arch import Local_Base


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return (
            gx,
            (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0),
            grad_output.sum(dim=3).sum(dim=2).sum(dim=0),
            None,
        )


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.0, reduce_channels=False):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(
            in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True
        )
        self.conv2 = nn.Conv2d(
            in_channels=dw_channel,
            out_channels=dw_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=dw_channel,
            bias=True,
        )
        self.conv3 = nn.Conv2d(
            in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True
        )

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=dw_channel // 2,
                out_channels=dw_channel // 2,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True,
            ),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(
            in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True
        )
        self.conv5 = nn.Conv2d(
            in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True
        )

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.tail = None
        if reduce_channels:
            self.tail = nn.Conv2d(c, c // 2, kernel_size=1, stride=1)

    def forward(self, input):
        x = input

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = input + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        out = y + x * self.gamma
        if self.tail is not None:
            return self.tail(out)
        return out


class NAFNet(nn.Module):
    def __init__(
        self,
        img_channel=3,
        layer_width=16,
        middle_blk_num=1,
        enc_blk_nums=[],
        dec_blk_nums=[],
        aid_blk_nums=[1, 1, 1, 1],
    ):
        super().__init__()

        self.intro = nn.Conv2d(
            in_channels=img_channel, out_channels=layer_width, kernel_size=3, padding=1, stride=1, groups=1, bias=True
        )
        self.ending = nn.Conv2d(
            in_channels=layer_width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True
        )

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        self.aid_heads = nn.ModuleList()
        self.aid_branchs = nn.ModuleList()

        chan = layer_width
        for i, num in enumerate(enc_blk_nums):
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, chan, 2, 2))

            self.aid_heads.append(Track1Head(in_channel=3 * 4 ** (i + 1), embed_dim=chan))
            self.aid_branchs.append(nn.Sequential(*[NAFBlock(2 * chan) for _ in range(1)]))

            chan = chan * 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blk_num)])

        for num in dec_blk_nums:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2)))
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))

        self.padder_size = 2 ** len(self.encoders)  # 2**4

    def down_sample(self, img):
        img1 = img[:, :, ::2, ::2]
        img2 = img[:, :, 1::2, 1::2]
        img3 = img[:, :, ::2, 1::2]
        img4 = img[:, :, 1::2, ::2]
        return torch.cat([img1, img2, img3, img4], dim=1)

    def forward(self, input, rgb):
        B, C, H, W = input.shape
        input = self.check_image_size_for_padding(input)
        rgb = self.check_image_size_for_padding(rgb)

        x = self.intro(input)

        encs = []

        for encoder, down, aid_head, aid_branch in zip(self.encoders, self.downs, self.aid_heads, self.aid_branchs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
            rgb = self.down_sample(rgb)
            x = aid_branch(torch.cat([x, aid_head(rgb)], dim=1))

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        # x = x + input
        return x[:, :, :H, :W]

    def check_image_size_for_padding(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


class NAFNetLocal(Local_Base, NAFNet):
    def __init__(self, *args, train_size=(1, 4, 256, 256), base_size=(640, 640), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        NAFNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        if base_size is None:
            base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if "module." in k else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


class Track1Head(nn.Module):
    def __init__(self, embed_dim, in_channel=1):
        super(Track1Head, self).__init__()
        self.head = nn.Sequential(
            ConvBNReLU2D(in_channel, out_channels=embed_dim, kernel_size=7, padding=3),
            NAFBlock(c=embed_dim),
            ConvBNReLU2D(embed_dim, out_channels=embed_dim, kernel_size=3, act="PReLU", padding=1),
        )

    def forward(self, input):
        return self.head(input)


class Track2Head(nn.Module):
    def __init__(self, embed_dim):
        super(Track2Head, self).__init__()
        self.rgb = nn.Sequential(
            ConvBNReLU2D(3, out_channels=embed_dim, kernel_size=7, padding=3),
            NAFBlock(c=embed_dim),
            ConvBNReLU2D(embed_dim, out_channels=embed_dim, kernel_size=3, act="PReLU", padding=1),
        )

        self.nir = nn.Sequential(
            ConvBNReLU2D(1, out_channels=embed_dim, kernel_size=7, padding=3),
            NAFBlock(c=embed_dim),
            ConvBNReLU2D(embed_dim, out_channels=embed_dim, kernel_size=3, act="PReLU", padding=1),
        )
        self.fuse = nn.Sequential(
            ConvBNReLU2D(embed_dim * 2 + 4, out_channels=embed_dim, kernel_size=3, padding=1),
            NAFBlock(c=embed_dim),
            ConvBNReLU2D(embed_dim, out_channels=embed_dim, kernel_size=3, act="PReLU", padding=1),
        )

    def forward(self, sample):
        nir, rgb = sample["lr_up"], sample["img_rgb"]
        # print(nir.shape, rgb.shape)
        out = torch.cat((self.rgb(rgb), self.nir(nir), rgb, nir), dim=1)
        return self.fuse(out)


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args

        self.head = Track2Head(args.embed_dim) if args.dataset == "NIR" else Track1Head(args.embed_dim)
        self.args = args
        enc_blks = [2, 2, 4, 8]
        middle_blk_num = 12
        dec_blks = [2, 2, 2, 2]
        train_size = (1, args.embed_dim, args.patch_size, args.patch_size)
        if args.test_only and args.tlc_enhance:
            self.net = NAFNetLocal(
                img_channel=args.embed_dim,
                layer_width=args.embed_dim,
                middle_blk_num=middle_blk_num,
                enc_blk_nums=enc_blks,
                dec_blk_nums=dec_blks,
                base_size=(640, 480),
                train_size=train_size,
            )
        else:
            self.net = NAFNet(
                img_channel=args.embed_dim,
                layer_width=args.embed_dim,
                middle_blk_num=middle_blk_num,
                enc_blk_nums=enc_blks,
                dec_blk_nums=dec_blks,
            )

        self.tail = ConvBNReLU2D(in_channels=args.embed_dim, out_channels=1, kernel_size=3, padding=1)

    def forward(self, samples):
        out = self.tail(self.net(self.head(samples), samples["img_rgb"]))
        # out = out if self.args.no_res else out + samples["lr_up"] # 是否残差
        return {"img_out": out if self.args.test_only else out}


def make_model(args):
    return Net(args)


if __name__ == "__main__":
    img_channel = 3
    width = 32

    # enc_blks = [2, 2, 4, 8]
    # middle_blk_num = 12
    # dec_blks = [2, 2, 2, 2]

    enc_blks = [1, 1, 1, 28]  # NAF在每个阶段的数量
    middle_blk_num = 1
    dec_blks = [1, 1, 1, 1]

    net = NAFNet(
        img_channel=img_channel,
        layer_width=width,
        middle_blk_num=middle_blk_num,
        enc_blk_nums=enc_blks,
        dec_blk_nums=dec_blks,
    )

    input_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, input_shape, verbose=False, print_per_layer_stat=True)

    params = float(params[:-3])
    macs = float(macs[:-4])

    print("mac, params: ", macs, params)
