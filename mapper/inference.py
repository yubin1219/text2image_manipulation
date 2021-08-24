import os
import sys
import time
from argparse import Namespace

import clip
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage

sys.path.append(".")
sys.path.append("..")

#from mapper.options.test_options import TestOptions
from mapper.styleclip_mapper import StyleCLIPMapper
