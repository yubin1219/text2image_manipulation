import os
import sys
import time
from argparse import Namespace
from PIL import Image
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
from mapper.latents_encoder.encoder import encoder

def inference_code(net, w, w_ori):
  device = "cuda" if torch.cuda.is_available() else 'cpu'
  with torch.no_grad():
    w_ori = w_ori.to(device).float()
    w = w.to(device).float()
    w_hat = w_ori + 0.1 * net.mapper(w)
    x_hat, w_ori = net.decoder([w_hat], input_is_latent=True, randomize_noise=True, return_latents=True, truncation=1)
  return x_hat, w_ori

def w_hat(w_ori, embeds, lookup_tensor, operation = 'sum'):
  em = embeds(lookup_tensor)
  m = torch.ones([18,1]).matmul(em.detach()).unsqueeze(0)
  if operation == 'sum':
    w = w_ori.cpu() + m
  elif operation == 'concat':
    w = torch.cat([m, w_ori.cpu()],dim = -1)
  return w

def w_hat_clip(w_ori, text_input, operation = 'concat', clip_model = None):
  device = "cuda" if torch.cuda.is_available() else 'cpu'
  with torch.no_grad():
    text_features = clip_model.encode_text(text_input)
  text_latents = torch.ones([18,1]).matmul(text_features.float().detach().cpu()).unsqueeze(0).to(device)
  if operation == 'sum':
    w = w_ori.to(device) + text_latents
  elif operation == 'concat':
    w = torch.cat([text_latents, w_ori], dim= -1)
  return w



def module_combine(test_opts):
  if test_opts.weights_download:
    ensure_checkpoint_exists("stylegan2-ffhq-config-f.pt")
    ensure_checkpoint_exists("color_sum.pt")
    ensure_checkpoint_exists("celeb_female_sum.pt")
    ensure_checkpoint_exists("celeb_male_sum.pt")
    ensure_checkpoint_exists("hairstyle_sum.pt")
    ensure_checkpoint_exists("color_cat.pt")
    ensure_checkpoint_exists("hairstyle_cat.pt")
    ensure_checkpoint_exists("color_clip.pt")
    ensure_checkpoint_exists("Disney_clip_cat.pt")
    ensure_checkpoint_exists("hairstyle_clip_cat.pt")
    
  device = "cuda" if torch.cuda.is_available() else 'cpu'
  
  opts_cat = {"stylegan_size" : 1024,
              "mapper_type": "LevelsMapper",
              "mapper_mode": "Mapper_cat",
              "no_coarse_mapper": False,
              "no_medium_mapper": False,
              "no_fine_mapper":False,
              "stylegan_weights" : 'stylegan2-ffhq-config-f.pt',
              "checkpoint_path" : None
             }
  opt_cat = Namespace(**opts_cat)
  net_cat = StyleCLIPMapper(opt_cat)
  net_cat.eval().to(device)

  opts_hair_cat = opts_cat.copy()
  opts_hair_cat["no_fine_mapper"] = True    # not change 'colors'
  opt_hair_cat = Namespace(**opts_hair_cat)
  net_hair_cat = StyleCLIPMapper(opt_hair_cat)
  net_hair_cat.eval().to(device)

  opts_hair_sum = opts_hair_cat.copy()
  opts_hair_sum["mapper_mode"] = "Mapper_sum"   # vector summation method
  opt_hair_sum = Namespace(**opts_hair_sum)
  net_hair_sum = StyleCLIPMapper(opt_hair_sum)
  net_hair_sum.eval().to(device)

  opts_sum = opts_hair_sum.copy()
  opts_sum["no_fine_mapper"] = False    # change 'colors'
  opt_sum = Namespace(**opts_sum)
  net_sum = StyleCLIPMapper(opt_sum)
  net_sum.eval().to(device)

  clip_model, _ = clip.load("ViT-B/32",device = device)
  clip_model.eval().to(device)

  images = []
  #modules = test_opts.modules.split(',')
  #texts = test_opts.texts.split(',')
  modules = test_opts.modules
  texts = test_opts.texts

  color_to_idx = {"blonde" : 0, "pink" : 1, "blue" : 2, "black" : 3}
  female_celeb_to_idx = {"Elsa" : 0, "Anna" : 1, "Emma_Stone": 2, "Anne_Hathaway": 3 , "Scarlett_Johansson": 4}
  male_celeb_to_idx = {"Ironman" : 0, "Dicaprio" : 1, "Zuckerberg": 2, "Tom_Holland": 3}
  hair_to_idx = {"curly" : 0, "wavy" : 1, "long": 2, "bobcut": 3 , "bangs": 4}

  # If use CLIP text encoder
  color = {"blonde" : "blonde hair", "red": "red hair", "pink" : "pink hair", "blue" : "blue hair", "purple" : "purple hair", "brown" : "brown hair", "black" : "black hair"}
  Disney = {"Elsa":"Elsa from Frozen", "Anna":"Anna from Frozen", "Rapunzel":"Rapunzel, Disney princess", "Ariel":"Ariel from the little mermaid, Disney princess"}
  hairstyle = {"wavy":"wavy hair", "long":"long hair", "bangs":"Bangs hair", "bobcut":"Bob-cut hairstyle"}

  
  if test_opts.new_latent:
    ensure_checkpoint_exists("e4e_ffhq_encode.pt")
    ensure_checkpoint_exists("shape_predictor_68_face_landmarks.dat")
    w_ori = encoder(test_opts.new_image_path)
    
  else:
    ensure_checkpoint_exists(test_opts.latent_path)
    latent = torch.load(test_opts.latent_path, map_location = device)
    w_ori = latent[test_opts.w_num].unsqueeze(0)

  with torch.no_grad():
    w_ori = w_ori.to(device)
    x, w_ori = net_cat.decoder([w_ori], input_is_latent=True, randomize_noise=True, return_latents=True, truncation=1)

  images.append(x)

  start = time.time()
  for i in range(len(modules)):
    if modules[i] == "celeb_female":
      net_sum.load_state_dict(torch.load("celeb_female_sum.pt", map_location = device)["state_dict"])
      torch.manual_seed(1)
      embeds_celebf = nn.Embedding(5,512)
      lookup_tensor = torch.tensor([female_celeb_to_idx[texts[i]]],dtype=torch.long)
      w = w_hat(w_ori, embeds_celebf, lookup_tensor, 'sum')
      x_hat, w_ori = inference_code(net_sum, w, w_ori)

    elif modules[i] == "celeb_male":
      net_sum.load_state_dict(torch.load("celeb_male_sum.pt", map_location = device)["state_dict"])
      torch.manual_seed(1)
      embeds_celebm = nn.Embedding(4,512)    
      lookup_tensor = torch.tensor([male_celeb_to_idx[texts[i]]],dtype=torch.long)
      w = w_hat(w_ori, embeds_celebm, lookup_tensor, 'sum')
      x_hat, w_ori = inference_code(net_sum, w, w_ori)

    elif modules[i] == "hair_sum":
      net_hair_sum.load_state_dict(torch.load("hairstyle_sum.pt", map_location = device)["state_dict"])
      torch.manual_seed(1)
      embeds_hair = nn.Embedding(5,512)    
      lookup_tensor = torch.tensor([hair_to_idx[texts[i]]],dtype=torch.long)
      w = w_hat(w_ori, embeds_hair, lookup_tensor, 'sum')
      x_hat, w_ori = inference_code(net_hair_sum, w, w_ori)

    elif modules[i] == "color_sum":
      net_sum.load_state_dict(torch.load("color_sum.pt", map_location = device)["state_dict"])
      torch.manual_seed(1)
      embeds_color = nn.Embedding(4,512)    
      lookup_tensor = torch.tensor([color_to_idx[texts[i]]],dtype=torch.long)
      w = w_hat(w_ori, embeds_color, lookup_tensor, 'sum')
      x_hat, w_ori = inference_code(net_sum, w, w_ori)

    elif modules[i] == "color_cat":
      net_cat.load_state_dict(torch.load("color_cat.pt", map_location = device)["state_dict"])
      torch.manual_seed(1)
      embeds_color = nn.Embedding(4,512)    
      lookup_tensor = torch.tensor([color_to_idx[texts[i]]],dtype=torch.long)
      w = w_hat(w_ori, embeds_color, lookup_tensor, 'concat')
      x_hat, w_ori = inference_code(net_cat, w, w_ori)
  
    elif modules[i] == "hair_cat":
      net_hair_cat.load_state_dict(torch.load("hairstyle_cat.pt", map_location = device)["state_dict"])
      torch.manual_seed(1)
      embeds_hair = nn.Embedding(5,512)    
      lookup_tensor = torch.tensor([hair_to_idx[texts[i]]],dtype=torch.long)
      w = w_hat(w_ori, embeds_hair, lookup_tensor, 'concat')
      x_hat, w_ori = inference_code(net_hair_cat, w, w_ori)

    elif modules[i] == "color_clip":
      net_cat.load_state_dict(torch.load("color_clip.pt", map_location = device)["state_dict"])
      text_input = torch.cat([clip.tokenize(color[texts[i]])]).to(device)
      w = w_hat_clip(w_ori, text_input, 'concat', clip_model)
      x_hat, w_ori = inference_code(net_cat, w, w_ori)

    elif modules[i] == "Disney_clip":
      net_cat.load_state_dict(torch.load("Disney_clip_cat.pt", map_location = device)["state_dict"])
      text_input = torch.cat([clip.tokenize(Disney[texts[i]])]).to(device)
      w = w_hat_clip(w_ori, text_input, 'concat', clip_model)
      x_hat, w_ori = inference_code(net_cat, w, w_ori)

    elif modules[i] == "hair_clip":
      net_hair_cat.load_state_dict(torch.load("hairstyle_clip_cat.pt", map_location = device)["state_dict"])
      text_input = torch.cat([clip.tokenize(hairstyle[texts[i]])]).to(device)
      w = w_hat_clip(w_ori, text_input, 'concat', clip_model)
      x_hat, w_ori = inference_code(net_hair_cat, w, w_ori)

    if test_opts.intermediate_outputs:
      images.append(x_hat)
    else:
      if i == (len(modules)-1) :
        images.append(x_hat)

  end = time.time()
  print("time : %.4f" %(end - start))

  result = torch.cat(images)
  image = ToPILImage()(make_grid(result.detach().cpu(), normalize=True, scale_each=True, value_range=(-1, 1), padding=0))
  h, w = image.size
  return image.resize((h // 2, w // 2))

if __name__ == '__main__':
  """
  celeb_female = ["Elsa", "Anna", "Emma_Stone", "Anne_Hathaway", "Scarlett_Johansson"]
  celeb_male = ["Ironman", "Dicaprio", "Zuckerberg", "Tom_Holland"]
  hair_sum or hair_cat = ["curly", "wavy", "long", "bobcut" , "bangs"]
  color_sum or hair_cat = ["blonde", "pink", "blue", "black"]
  
  Disney_clip = ["Elsa", "Anna", "Rapunzel", "Ariel"]
  hair_clip = ["wavy", "long", "bangs", "bobcut"]
  color_clip = ["blonde", "red", "pink", "blue", "purple", "brown", "black"]
  """
  test_options = {"exp_dir": "results/",
                  "new_latent": False,
                  "new_image_path": None,
                  "weights_download": True,
                  "latent_path": "test_female.pt",
                  "intermediate_outputs": True,
                  "w_num": 60,
                  "modules": ["celeb_female","hair_sum","color_sum"], # "celeb_female", "celeb_male", "hair_sum", "color_sum" / "hair_cat", "color_cat" / "color_clip" , "hair_clip", "Disney_clip"
                  "texts": ["Emma_Stone","wavy", "blonde"]}
  #test_opts = Namespace(**test_options)
  test_opts = TestOptions().parse()
  images = module_combine(test_opts)
  images.save("results_{}.png".format(test_opts.texts))
