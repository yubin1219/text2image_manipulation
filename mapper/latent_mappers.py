import torch
from torch import nn
from torch.nn import Module
from models.stylegan2.model import EqualLinear, PixelNorm

class Mapper(Module):
    def __init__(self, opts):
        super(Mapper, self).__init__()
        self.opts = opts
        layers = [PixelNorm()]
        
        for i in range(4):
            layers.append(
                EqualLinear(
                    512, 512, lr_mul=0.01, activation='fused_lrelu' # vector size
                )
            )
        
        self.mapping = nn.Sequential(*layers)


    def forward(self, x):
        x = self.mapping(x)
        return x


class Mapper_cat(Module):
    def __init__(self, opts):
        super(Mapper_cat, self).__init__()
        self.opts = opts
        layers = [PixelNorm()]
        
        for i in range(7):
            layers.append(
                EqualLinear(
                    1024, 1024, lr_mul=0.01, activation='fused_lrelu' # vector size
                )
            )
        layers.append(
                EqualLinear(
                    1024, 512, lr_mul=0.01, activation='fused_lrelu' # vector size
                )
        )
        
        self.mapping = nn.Sequential(*layers)


    def forward(self, x):
        x = self.mapping(x)
        return x
    
class Mapper_sum(Module):
    def __init__(self, opts):
        super(Mapper_sum, self).__init__()
        self.opts = opts
        layers = [PixelNorm()]
        
        for i in range(8):
            layers.append(
                EqualLinear(
                    512, 512, lr_mul=0.01, activation='fused_lrelu' # vector size
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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.opts = opts

        if not opts.no_coarse_mapper:
            if opts.mapper_mode == "Mapper_cat":
                self.course_mapping = Mapper_cat(opts)
            elif opts.mapper_mode == "Mapper_sum":
                self.course_mapping = Mapper_sum(opts)
                
        if not opts.no_medium_mapper:
            if opts.mapper_mode == "Mapper_cat":
                self.medium_mapping = Mapper_cat(opts)
            elif opts.mapper_mode == "Mapper_sum":
                self.medium_mapping = Mapper_sum(opts)
                
        if not opts.no_fine_mapper:
            if opts.mapper_mode == "Mapper_cat":
                self.fine_mapping = Mapper_cat(opts)
            elif opts.mapper_mode == "Mapper_sum":
                self.fine_mapping = Mapper_sum(opts)
            
    def forward(self, x):
        s1,s2,s3 = x.size()
        x_coarse = x[:, :4, :]
        x_medium = x[:, 4:8, :]
        x_fine = x[:, 8:, :]

        if not self.opts.no_coarse_mapper:
            x_coarse = self.course_mapping(x_coarse)
        else:
            x_coarse = torch.zeros([s1,4,512]).to(self.device)
            
        if not self.opts.no_medium_mapper:
            x_medium = self.medium_mapping(x_medium)
        else:
            x_medium = torch.zeros([s1,4,512]).to(self.device)
            
        if not self.opts.no_fine_mapper:
            x_fine = self.fine_mapping(x_fine)
        else:
            x_fine = torch.zeros([s1,10,512]).to(self.device)


        out = torch.cat([x_coarse, x_medium, x_fine], dim=1)

        return out
