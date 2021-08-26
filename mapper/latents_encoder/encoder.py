from argparse import Namespace
import sys
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
import dlib

sys.path.append(".")
sys.path.append("..")

from mapper.latents_encoder.common import tensor2im
from mapper.latents_encoder.psp import pSp
from mapper.latents_encoder.alignment import align_face

def run_alignment(image_path):
  predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
  aligned_image = align_face(filepath=image_path, 
                             predictor=predictor) 
  print("Aligned image has shape: {}".format(aligned_image.size))
  return aligned_image 

def display_alongside_source_image(result_image, source_image):
  res = np.concatenate([np.array(source_image.resize((256,256))),
                        np.array(result_image.resize((256,256)))], axis=1)
  return Image.fromarray(res)

def run_on_batch(inputs, net):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  images, latents = net(inputs.to(device).float(), randomize_noise=False, return_latents=True)
  return images, latents


def encoder(new_img_path):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  
  model_path = "e4e_ffhq_encode.pt"
  ckpt = torch.load(model_path, map_location=device)
  opts = ckpt['opts']
  opts['checkpoint_path'] = model_path
  opts[device] = device
  opts = Namespace(**opts)
  net = pSp(opts)
  net.eval().to(device)
  
  image_path = new_img_path
  original_image = Image.open(image_path)
  original_image = original_image.convert("RGB")
  input_image = run_alignment(image_path)
  input_image.resize((256,256))
  
  img_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
  
  transformed_image = img_transforms(input_image)
  with torch.no_grad():
    images, latents = run_on_batch(transformed_image.unsqueeze(0), net)
    result_image, latent = images[0], latents[0]
  torch.save(latents, 'latents.pt')
  
  # Display inversion:
  display_alongside_source_image(tensor2im(result_image), input_image)
  
  return latents
