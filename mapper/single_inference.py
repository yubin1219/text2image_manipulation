import os
import sys
import time
from argparse import Namespace
import matplotlib.pyplot as plt

import clip
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage

sys.path.append(".")
sys.path.append("..")

from mapper.options.test_options import TestOptions
from mapper.utils import ensure_checkpoint_exists
from mapper.styleclip_mapper import StyleCLIPMapper
from mapper.inference import inference_code

def single_inference(test_opts):
  if test_opts.weights_download:
    ensure_checkpoint_exists("multi_sum.pt")

  device = "cuda" if torch.cuda.is_available() else 'cpu'

  opts = {"stylegan_size" : 1024,
         "mapper_type": "LevelsMapper",
         "mapper_mode": "Mapper_multi",
         "no_coarse_mapper": False,
         "no_medium_mapper": False,
         "no_fine_mapper":False,
         "checkpoint_path" : "multi_sum.pt"
          }
  opt = Namespace(**opts)
  net = StyleCLIPMapper(opt)
  net.eval().to(device)

  torch.manual_seed(1)
  class_to_idx = {"Elsa" : 0, "Emma_Watson": 1, "wavy" : 2, "bangs" : 3, "pink": 4,"black": 5}

  embeds = nn.Embedding(6, 512)
  
  images = []
  texts = test_opts.texts

  ensure_checkpoint_exists(test_opts.latent_path)
  latent = torch.load(test_opts.latent_path, map_location = device)
  w_ori = latent[test_opts.w_num].unsqueeze(0)

  with torch.no_grad():
    w_ori = w_ori.to(device)
    x, w_ori = net.decoder([w_ori], input_is_latent=True, randomize_noise=True, return_latents=True, truncation=1)
  
  images.append(x)

  start = time.time()
  em = 0
  for t in texts:
    lookup_tensor = torch.tensor([class_to_idx[t]],dtype=torch.long)
    em += embeds(lookup_tensor)

  m = torch.ones([18,1]).matmul(em.detach()).unsqueeze(0)
  w = w_ori.cpu() + m
  x_hat, w_ori = inference_code(net, w, w_ori)

  images.append(x_hat)

  end = time.time()
  print("time : %.4f" %(end - start))

  result = torch.cat(images)
  image = ToPILImage()(make_grid(result.detach().cpu(), normalize=True, scale_each=True, value_range=(-1, 1), padding=0))
  h, w = image.size
  return image.resize((h // 2, w // 2))

if __name__=="__main__":
  test_options = {"weights_download": True,
                "latent_path": "test_female.pt",
                "w_num": 60,
                "texts": ["wavy","black"]   # ["Elsa" , "Emma_Watson", "wavy", "bangs", "pink", "black"]
                }
  #test_opts = Namespace(**test_options)
  test_opts = TestOptions().parse()
  images = single_inference(test_opts)  
  images.save("results_{}.png".format(test_opts.texts))
  plt.figure(figsize=(16,16))
  plt.imshow(images)
  plt.axis('off')
  plt.show()
