from torch.utils.data import Dataset

class LatentsDataset(Dataset):
	def __init__(self, latents, opts):
		self.latents = latents
		self.opts = opts

	def __len__(self):
		return self.latents.shape[0]

	def __getitem__(self, index):
		w = self.latents[index]
    		if w[0][0]==1:
      			t = "blonde hair"
    		elif w[0][1]==1:
      			t= "pink hair"
    		elif w[0][2]==1:
      			t = "blue hair"
    		elif w[0][3]==1:
     		 	t = "black hair"
    		return [w,t]

