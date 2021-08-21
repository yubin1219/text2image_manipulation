"""
LatentsDataset : get text embedding vector from nn.embedding
LatentsDataset_clip : get text embedding vector from CLIP text encoder
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import torch
from torch.utils.data import Dataset
import torch.nn as nn

class LatentDataset(Dataset):
   def __init__(self, latents, opts, dataset_mode):
      self.latents = latents
      self.opts = opts
      self.dataset_mode = dataset_mode
    
   def __len__(self):
      return self.latents.shape[0]
    
   def __getitem__(self, index):
      torch.manual_seed(1)
      w_ori = self.latents[index]
      
      if self.dataset_mode == "train":
         dataset_size = self.opts.train_dataset_size
      elif self.dataset_mode == "test":
         dataset_size = self.opts.test_dataset_size
         
      if self.opts.data_mode == "hair":
         hair_to_idx = {"Curly": 0, "Wavy": 1, "Long": 2, "Bobcut": 3, "Bangs": 4}
         embeds = nn.Embedding(5,512)

         lookup_tensor1 = torch.tensor([hair_to_idx["Curly"]], dtype = torch.long)
         lookup_tensor2 = torch.tensor([hair_to_idx["Wavy"]], dtype = torch.long)
         lookup_tensor3 = torch.tensor([hair_to_idx["Long"]], dtype = torch.long)
         lookup_tensor4 = torch.tensor([hair_to_idx["Bobcut"]], dtype = torch.long)
         lookup_tensor5 = torch.tensor([hair_to_idx["Bangs"]], dtype = torch.long)

         em1 = embeds(lookup_tensor1)
         em2 = embeds(lookup_tensor2)
         em3 = embeds(lookup_tensor3)
         em4 = embeds(lookup_tensor4)
         em5 = embeds(lookup_tensor5)
      
         if index < int(dataset_size / 5):
            t = "Curly hair"
            m = torch.ones([18,1]).matmul(em1.detach())
         elif int(dataset_size / 5) <= index < int( 2 * dataset_size / 5):
            t = "Wavy hair"
            m = torch.ones([18,1]).matmul(em2.detach())
         elif int(2 * dataset_size / 5) <= index < int(3 * dataset_size / 5):
            t = "Long hair"
            m = torch.ones([18,1]).matmul(em3.detach())
         elif int(3 * dataset_size / 5) <= index < int(4 * dataset_size / 5):
            t = "Bob-cut hairstyle"
            m = torch.ones([18,1]).matmul(em4.detach())
         elif int(4 * dataset_size / 5) <= index < int(dataset_size):
            t = "Bangs hair"
            m = torch.ones([18,1]).matmul(em5.detach())
      
      elif self.opts.data_mode == "female":
         female_to_idx = {"Elsa": 0, "Anna": 1, "Emma Stone": 2, "Anne Hathaway": 3, "Scarlett Johansson": 4}
         embeds = nn.Embedding(5,512)

         lookup_tensor1 = torch.tensor([female_to_idx["Elsa"]], dtype = torch.long)
         lookup_tensor2 = torch.tensor([female_to_idx["Anna"]], dtype = torch.long)
         lookup_tensor3 = torch.tensor([female_to_idx["Emma Stone"]], dtype = torch.long)
         lookup_tensor4 = torch.tensor([female_to_idx["Anne Hathaway"]], dtype = torch.long)
         lookup_tensor5 = torch.tensor([female_to_idx["Scarlett Johansson"]], dtype = torch.long)

         em1 = embeds(lookup_tensor1)
         em2 = embeds(lookup_tensor2)
         em3 = embeds(lookup_tensor3)
         em4 = embeds(lookup_tensor4)
         em5 = embeds(lookup_tensor5)
      
         if index < int(dataset_size/5):
            t = "Elsa from Frozen"
            m = torch.ones([18,1]).matmul(em1.detach())
         elif int(dataset_size / 5) <= index < int( 2 * dataset_size / 5):
            t = "Anna from Frozen"
            m = torch.ones([18,1]).matmul(em2.detach())
         elif int(2 * dataset_size / 5) <= index < int( 3 * dataset_size / 5):
            t = "Emma Stone"
            m = torch.ones([18,1]).matmul(em3.detach())
         elif int(3 * dataset_size / 5) <= index < int( 4 * dataset_size / 5):
            t = "Anne Hathaway"
            m = torch.ones([18,1]).matmul(em4.detach())
         elif int(4 * dataset_size / 5) <= index < int(dataset_size):
            t = "Scarlett Johansson"
            m = torch.ones([18,1]).matmul(em5.detach())
      
      elif self.opts.data_mode == "male":
         male_to_idx = {"Iron man": 0, "Dicaprio": 1, "Zuckerberg": 2, "Tom Holland": 3}
         embeds = nn.Embedding(4,512)

         lookup_tensor1 = torch.tensor([male_to_idx["Iron man"]], dtype = torch.long)
         lookup_tensor2 = torch.tensor([male_to_idx["Dicaprio"]], dtype = torch.long)
         lookup_tensor3 = torch.tensor([male_to_idx["Zuckerberg"]], dtype = torch.long)
         lookup_tensor4 = torch.tensor([male_to_idx["Tom Holland"]], dtype = torch.long)

         em1 = embeds(lookup_tensor1)
         em2 = embeds(lookup_tensor2)
         em3 = embeds(lookup_tensor3)
         em4 = embeds(lookup_tensor4)
      
         if index < int(dataset_size/4):
            t = "Robert Downey Jr., Actor of Iron man"
            m = torch.ones([18,1]).matmul(em1.detach())
         elif int(dataset_size / 4) <= index < int( 2 *dataset_size / 4):
            t = "Leonardo Wilhelm Dicaprio"
            m = torch.ones([18,1]).matmul(em2.detach())
         elif int(2 * dataset_size / 4) <= index < int( 3 * dataset_size / 4):
            t = "Mark Zuckerberg"
            m = torch.ones([18,1]).matmul(em3.detach())
         elif int(3 * dataset_size / 4) <= index < int(dataset_size):
            t = "Tom Holland, Actor of Spiderman"
            m = torch.ones([18,1]).matmul(em4.detach())
      
      elif self.opts.data_mode == "color":
         color_to_idx = {"blonde" : 0, "pink": 1, "blue": 2, "black": 3}
         embeds = nn.Embedding(4,512)

         lookup_tensor1 = torch.tensor([color_to_idx["blonde"]], dtype = torch.long)
         lookup_tensor2 = torch.tensor([color_to_idx["pink"]], dtype = torch.long)
         lookup_tensor3 = torch.tensor([color_to_idx["blue"]], dtype = torch.long)
         lookup_tensor4 = torch.tensor([color_to_idx["black"]], dtype = torch.long)

         em1 = embeds(lookup_tensor1)
         em2 = embeds(lookup_tensor2)
         em3 = embeds(lookup_tensor3)
         em4 = embeds(lookup_tensor4)
      
         if index < int(dataset_size/4):
            t = "blonde hair"
            m = torch.ones([18,1]).matmul(em1.detach())
         elif int(dataset_size / 4) <= index < int(2 * dataset_size/4):
            t = "pink hair"
            m = torch.ones([18,1]).matmul(em2.detach())
         elif int(2 * dataset_size / 4) <= index < int(3 * dataset_size/4):
            t = "blue hair"
            m = torch.ones([18,1]).matmul(em3.detach())
         elif int(3 * dataset_size / 4) <= index < int(dataset_size):
            t = "black hair"
            m = torch.ones([18,1]).matmul(em4.detach())
      
      elif self.opts.data_mode == "multi":
         multi_to_idx = {"Elsa": 0, "Emma Watson": 1, "wavy": 2, "bangs": 3, "pink": 4, "black": 5}
         embeds = nn.Embedding(6,512)

         lookup_tensor1 = torch.tensor([multi_to_idx["Elsa"]], dtype = torch.long)
         lookup_tensor2 = torch.tensor([multi_to_idx["Emma Watson"]], dtype = torch.long)
         lookup_tensor3 = torch.tensor([multi_to_idx["wavy"]], dtype = torch.long)
         lookup_tensor4 = torch.tensor([multi_to_idx["bangs"]], dtype = torch.long)
         lookup_tensor5 = torch.tensor([multi_to_idx["pink"]], dtype = torch.long)
         lookup_tensor6 = torch.tensor([multi_to_idx["black"]], dtype = torch.long)

         em1 = embeds(lookup_tensor1)
         em2 = embeds(lookup_tensor2)
         em3 = embeds(lookup_tensor3)
         em4 = embeds(lookup_tensor4)
         em5 = embeds(lookup_tensor5)
         em6 = embeds(lookup_tensor6)
      
         if index < int(dataset_size / 26) :
            t = ["Elsa from Frozen"]
            m = torch.ones([18,1]).matmul(em1.detach())

         elif int(dataset_size / 26) <= index < int(2 * dataset_size / 26):
            t = ["Emma Watson, Actress"]
            m = torch.ones([18,1]).matmul(em2.detach())

         elif int(2 * dataset_size / 26) <= index < int(3 * dataset_size / 26):
            t = ["wavy hair"]
            m = torch.ones([18,1]).matmul(em3.detach())

         elif int(3 *dataset_size / 26) <= index < int(4 * dataset_size / 26):
            t = ["bangs hair"]
            m = torch.ones([18,1]).matmul(em4.detach())

         elif int(4 * dataset_size / 26) <= index < int(5 *dataset_size / 26):
            t = ["pink hair"]
            m = torch.ones([18,1]).matmul(em5.detach())

         elif int(5 * dataset_size / 26) <= index < int(6 * dataset_size / 26):
            t = ["black hair"]
            m = torch.ones([18,1]).matmul(em6.detach())

         elif int(6 * dataset_size / 26) <= index < int(7 * dataset_size / 26):
            t = ["Elsa from Frozen","wavy hair"]
            em = em1 + em3
            m = torch.ones([18,1]).matmul(em.detach())

         elif int(7 *dataset_size / 26) <= index < int(8 * dataset_size / 26):
            t = ["Elsa from Frozen","bangs hair"]
            em = em1 + em4
            m = torch.ones([18,1]).matmul(em.detach())
    
         elif int(8*dataset_size / 26) <= index < int(9 *dataset_size / 26):
            t = ["Elsa from Frozen","pink hair"]
            em = em1 + em5
            m = torch.ones([18,1]).matmul(em.detach())

         elif int(9*dataset_size / 26) <= index < int(10 * dataset_size / 26):
            t = ["Elsa from Frozen","black hair"]
            em = em1 + em6
            m = torch.ones([18,1]).matmul(em.detach())

         elif int(10*dataset_size / 26) <= index < int(11 *dataset_size / 26):
            t = ["Emma Watson, Actress","wavy hair"]
            em = em2 + em3
            m = torch.ones([18,1]).matmul(em.detach())

         elif int(11*dataset_size / 26) <= index < int(12 * dataset_size / 26):
            t = ["Emma Watson, Actress","bangs hair"]
            em = em2 + em4
            m = torch.ones([18,1]).matmul(em.detach())

         elif int(12*dataset_size / 26) <= index < int(13 * dataset_size / 26):
            t = ["Emma Watson, Actress","pink hair"]
            em = em2 + em5
            m = torch.ones([18,1]).matmul(em.detach())

         elif int(13*dataset_size / 26) <= index < int(14 * dataset_size / 26):
            t = ["Emma Watson, Actress","black hair"]
            em = em2 + em6
            m = torch.ones([18,1]).matmul(em.detach())

         elif int(14*dataset_size / 26) <= index < int(15 * dataset_size / 26):
            t = ["wavy hair","pink hair"]
            em = em3 + em5
            m = torch.ones([18,1]).matmul(em.detach())

         elif int(15*dataset_size / 26) <= index < int(16 * dataset_size / 26):
            t = ["wavy hair","black hair"]
            em = em3 + em6
            m = torch.ones([18,1]).matmul(em.detach())
    
         elif int(16*dataset_size / 26) <= index < int(17 * dataset_size / 26):
            t = ["bangs hair","black hair"]
            em = em4 + em6
            m = torch.ones([18,1]).matmul(em.detach())

         elif int(17*dataset_size / 26) <= index < int(18 * dataset_size / 26):
            t = ["bangs hair","pink hair"]
            em = em4 + em5
            m = torch.ones([18,1]).matmul(em.detach())

         elif int(18*dataset_size / 26) <= index < int(19 *dataset_size / 26):
            t = ["Elsa from Frozen","wavy hair","pink hair"]
            em = em1 + em3 + em5
            m = torch.ones([18,1]).matmul(em.detach())

         elif int(19*dataset_size / 26) <= index < int(20 * dataset_size / 26):
            t = ["Elsa from Frozen","wavy hair","black hair"]
            em = em1 + em3 + em6
            m = torch.ones([18,1]).matmul(em.detach())

         elif int(20*dataset_size / 26) <= index < int(21 * dataset_size / 26):
            t = ["Elsa from Frozen","bangs hair","pink hair"]
            em = em1 + em4 + em5
            m = torch.ones([18,1]).matmul(em.detach())

         elif int(21*dataset_size / 26) <= index < int(22 * dataset_size / 26):
            t = ["Elsa from Frozen","bangs hair","black hair"]
            em = em1 + em4 + em6
            m = torch.ones([18,1]).matmul(em.detach())

         elif int(22*dataset_size / 26) <= index < int(23 * dataset_size / 26):
            t = ["Emma Watson, Atress","wavy hair","pink hair"]
            em = em2 + em3 + em5
            m = torch.ones([18,1]).matmul(em.detach())

         elif int(23*dataset_size / 26) <= index < int(24 * dataset_size / 26):
            t = ["Emma Watson, Actress","wavy hair","black hair"]
            em = em2 + em3 + em6
            m = torch.ones([18,1]).matmul(em.detach())

         elif int(24*dataset_size / 26) <= index < int(25 * dataset_size / 26):
            t = ["Emma Watson, Actress","bangs hair","pink hair"]
            em = em2 + em4 + em5
            m = torch.ones([18,1]).matmul(em.detach())

         elif int(25*dataset_size / 26) <= index < int(dataset_size):
            t = ["Emma Watson, Actress","bangs hair","black hair"]
            em = em2 + em4 + em6
            m = torch.ones([18,1]).matmul(em.detach())
      
      if self.opts.mapper_mode == "Mapper_sum":
         w = w_ori + m
      elif self.opts.mapper_mode == "Mapper_cat":
         w = torch.cat([m, w_ori], dim = -1)
      
      return [w, w_ori, t]
  

class LatentDataset_clip(Dataset):
   def __init__(self, latents, opts, dataset_mode):
      self.latents = latents
      self.opts = opts
      self.dataset_mode = dataset_mode
    
   def __len__(self):
      return self.latents.shape[0]
    
   def __getitem__(self, index):
      w_ori = self.latents[index]
      if self.dataset_mode == "train":
         dataset_size = self.opts.train_dataset_size
      elif self.dataset_mode == "test":
         dataset_size = self.opts.test_dataset_size
      
      if self.opts.data_mode == "Disney":
         if index < int(dataset_size/4):
            t = "Elsa from Frozen"
         elif int(dataset_size/4) <= index < int(2*dataset_size/4):
            t = "Anna from Frozen"
         elif int(2*dataset_size/4) <= index < int(3*dataset_size/4):
            t = "Rapunzel, Disney princess"
         elif int(3*dataset_size/4) <= index < int(dataset_size):
            t = "Ariel from the little mermaid, Disney princess"
        
      elif self.opts.data_mode == "color":
         if index < int(dataset_size/6):
            t = "blonde hair"
         elif int(dataset_size/6) <= index < int(2*dataset_size/6):
            t = "red hair"
         elif int(2*dataset_size/6) <= index < int(3*dataset_size/6):
            t = "pink hair"
         elif int(3*dataset_size/6) <= index < int(4*dataset_size/6):
            t = "blue hair"
         elif int(4*dataset_size/6) <= index < int(5*dataset_size/6):
            t = "purple hair"
         elif int(5*dataset_size/6) <= index < int(dataset_size):
            t = "black hair"
        
      elif self.opts.data_mode == "hair":
         if index < int(dataset_size/4):
            t = "wavy hair"
         elif int(dataset_size/4) <= index < int(2*dataset_size/4):
            t = "Bangs hair"
         elif int(2*dataset_size/4) <= index < int(3*dataset_size/4):
            t = "Bob-cut hairstyle"
         elif int(3*dataset_size/4) <= index < int(dataset_size):
            t = "Long hair"
        
      return [w_ori, t]
