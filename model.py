import torch
import logging
import torch.nn as nn
from blocks import Focus
from blocks import ConvBNSiLU
from blocks import C3
from blocks import Bottleneck
from blocks import SPPF
from blocks import Concat


anchors = [
    [(10, 13), (16, 30), (33, 23)], # P3/8
    [(30, 61), (62, 45), (59, 119)], # P4/16
    [(116, 90), (156, 198), (373, 326)] # P5/32
]

nc = 10




class Detect(nn.Module):
    stride = None
    dynamic = False
    export = False
    def __init__(self, anchors, nc=20, ch=()):
        super(Detect, self).__init__()
        self.nc = nc # number of classes
        self.no = nc + 5 # number of output per anchor
        self.nl = len(anchors) # number of detection layers
        self.na = len(anchors[0]) # number of anchors
        anchors_ = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, kernel_size=1, stride=1, padding=0) for x in ch)

    def forward(self, x):
        
        # x shape: (N, 80 x 80 x 256) / (N, 40 x 40 x 512) / (N, 20 x 20 x 1024)
        for i in range(self.nl):
            x[i] = self.m[i](x[i]) # convolution
            batch_size, _, ny, nx = x[i].shape # x shape : (N, 255 x 80 x 80) / (N, 255 x 40 x 40) / (N, 255 x 20 x 20)
            x[i] = x[i].view(batch_size, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

        return x



architecture = [
    # from, number, module, args

    #  backbone
    [-1, 1, ConvBNSiLU, [64, 6, 2, 2]],
    [-1, 1, ConvBNSiLU, [128, 3, 2, 1]],
    [-1, 3, C3, [128, True]],
    [-1, 1, ConvBNSiLU, [256, 3, 2, 1]],
    [-1, 6, C3, [256, True]],
    [-1, 1, ConvBNSiLU, [512, 3, 2, 1]],
    [-1, 9, C3, [512, True]],
    [-1, 1, ConvBNSiLU, [1024, 3, 2, 1]],
    [-1, 3, C3, [1024, True]],
    [-1, 1, SPPF, [1024, 5]],

    # head
    [-1, 1, ConvBNSiLU, [512, 1, 1, 0]],
    [-1, 1, nn.Upsample, [2, "nearest"]],
    [[-1, 6], 1, Concat, [1]],
    [-1, 3, C3, [512, False]],
    [-1, 1, ConvBNSiLU, [256, 1, 1, 0]],
    [-1, 1, nn.Upsample, [2, "nearest"]],
    [[-1, 4], 1, Concat, [1]],
    [-1, 3, C3, [256, False]],
    [-1, 1, ConvBNSiLU, [256, 3, 2, 1]],
    [[-1, 14], 1, Concat, [1]],
    [-1, 3, C3, [512, False]],
    [-1, 1, ConvBNSiLU, [512, 3, 2, 1]],
    [[-1, 10], 1, Concat, [1]],
    [-1, 3, C3, [1024, False]],

    # detect
    [[17, 20, 23], 1, Detect, [nc, anchors]]
]


class YOLOv5(nn.Module):
    def __init__(self, in_channels=3):
        super(YOLOv5, self).__init__()
        self.in_channels = in_channels
        self.arch = architecture
        self.layers = self.parse_module(architecture)
        self.ch = architecture[-1][0]

    def forward(self, x):
        output, routes = [], []
        for i, layer in enumerate(self.layers):
            if i in self.ch:
                x = layer(x)
                output.append(x)
                routes.append(x)
                continue
                
            if isinstance(layer, Concat):
                f = self.arch[i][0][1]
                x = layer((x, routes[f]))
                routes.append(x)
                continue
            

            if isinstance(layer, Detect):
                x = layer(output)
                continue

            x = layer(x)
            routes.append(x)

        return x

    def parse_module(self, configurations):
        layers = nn.ModuleList()
        in_channels = self.in_channels
        try:
            for i, layer in enumerate(configurations):
                f, rep, module, args = layer
                if module is ConvBNSiLU:
                    out_channels, kernel_size, stride, padding = args
                    layers += [module(
                        in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding
                        )
                    ]
                    
                    in_channels = out_channels
                
                elif module is C3:
                    out_channels, shortcut = args
                    layers += [module(in_channels=in_channels, out_channels=out_channels, n=rep, shortcut=shortcut)]
                    in_channels = out_channels
                
                elif module is Concat:
                    layers += [
                        module(dimension=args[0])
                    ]
                    
                    in_channels = configurations[f[1]][3][0] + in_channels

                elif module is SPPF:
                    out_channels, kernel_sizes = args
                    layers += [module(in_channels=in_channels, out_channels=out_channels, kernel_sizes=[kernel_sizes] * 3)]
    
                elif module is nn.Upsample:
                    layers += [
                        module(scale_factor=args[0], mode=args[1])
                    ]

                elif module is Detect:
                    ch = (configurations[i][3][0] for i in f)
                    nc, anchors = args
                    layers += [module(anchors, nc, ch=ch)]
                else:
                    raise ValueError('Error: Undefined  module %s' %module)

        except Exception as err:
            logging.error('Unexpected Error while parsing %s layers: %s' %(module, err))    

        return layers



if __name__ == "__main__":
    x = torch.randn((2, 3, 480, 640))
    model = YOLOv5(in_channels=3)
    out = model(x)

    print(f"Scale 1: {out[0].shape}")
    print(f"Scale 2: {out[1].shape}")
    print(f"Scale 3: {out[2].shape}")
