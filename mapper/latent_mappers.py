import torch
from torch import nn
from torch.nn import Module
from models.stylegan2.model import EqualLinear, PixelNorm


class Mapper_color(Module):
    def __init__(self, opts):
        super(Mapper_color, self).__init__()
        self.mode = opts.mapper_mode
        self.opts = opts
        layers = [PixelNorm()]

        for i in range(7):
            layers.append(
                EqualLinear(
                    516, 516, lr_mul=0.01, activation='fused_lrelu' # vector size
                )
            )
        layers.append(
                EqualLinear(
                    516, 512, lr_mul=0.01, activation='fused_lrelu' # vector size
                )
        )
        
        self.mapping = nn.Sequential(*layers)


    def forward(self, x):
        x = self.mapping(x)
        return x

class Mapper_hair(Module):
    def __init__(self, opts):
        super(Mapper_hair, self).__init__()

        self.opts = opts
        layers = [PixelNorm()]

        for i in range(7):
            layers.append(
                EqualLinear(
                    517, 517, lr_mul=0.01, activation='fused_lrelu'
                )
            )
        layers.append(
                EqualLinear(
                    517, 512, lr_mul=0.01, activation='fused_lrelu' # vector size
                )
        )
        self.mapping = nn.Sequential(*layers)


    def forward(self, x):
        x = self.mapping(x)
        return x
    
class SingleMapper(Module):
    def __init__(self, opts):
        super(SingleMapper, self).__init__()

        self.opts = opts

        self.mapping = Mapper(opts)

    def forward(self, x):
        out = self.mapping(x)
        return out


class LevelsMapper(Module):
    def __init__(self, opts):
        super(LevelsMapper, self).__init__()
        self.mapper_mode = opts.mapper_mode
        self.opts = opts

        if not opts.no_coarse_mapper:
            if opts.mapper_mode == "hair":
                self.course_mapping = Mapper_hair(opts)
            elif opts.mapper_mode == "color":
                self.course_mapping = Mapper_color(opts)
                
        if not opts.no_medium_mapper:
            if opts.mapper_mode == "hair":
                self.course_mapping = Mapper_hair(opts)
            elif opts.mapper_mode == "color":
                self.course_mapping = Mapper_color(opts)
                
        if not opts.no_fine_mapper:
            if opts.mapper_mode == "hair":
                self.course_mapping = Mapper_hair(opts)
            elif opts.mapper_mode == "color":
                self.course_mapping = Mapper_color(opts)

    def forward(self, x):
        x_coarse = x[:, :4, :]
        x_medium = x[:, 4:8, :]
        x_fine = x[:, 8:, :]

        if not self.opts.no_coarse_mapper:
            x_coarse = self.course_mapping(x_coarse)
        else:
            x_coarse = torch.zeros_like(x_coarse)
            
        if not self.opts.no_medium_mapper:
            x_medium = self.medium_mapping(x_medium)
        else:
            x_medium = torch.zeros_like(x_medium)
            
        if not self.opts.no_fine_mapper:
            x_fine = self.fine_mapping(x_fine)
        else:
            x_fine = torch.zeros_like(x_fine)


        out = torch.cat([x_coarse, x_medium, x_fine], dim=1)

        return out
