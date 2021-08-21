import os

import clip
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import criteria.clip_loss as clip_loss
from criteria import id_loss
from mapper.dataset.latents_dataset import LatentDataset, LatentDataset_clip
from mapper.styleclip_mapper import StyleCLIPMapper
from mapper.training.ranger import Ranger
from mapper.training import train_utils


class Coach:
	def __init__(self, opts):
		self.opts = opts

		self.global_step = 0
		
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.opts.device = self.device
		
		if self.opts.text_embed_mode == "clip_encoder":
			self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
			
		self.net = StyleCLIPMapper(self.opts).to(self.device)

		# Initialize loss
		if self.opts.id_lambda > 0:
			self.id_loss = id_loss.IDLoss(self.opts).to(self.device).eval()
		if self.opts.clip_lambda > 0:
			self.clip_loss = clip_loss.CLIPLoss(opts)
		if self.opts.latent_l2_lambda > 0:
			self.latent_l2_loss = nn.MSELoss().to(self.device).eval()

		# Initialize optimizer
		self.optimizer = self.configure_optimizers()

		# Initialize dataset
		self.train_dataset, self.test_dataset = self.configure_datasets()
		self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.opts.batch_size, shuffle=True,
						   num_workers=int(self.opts.workers), drop_last=True)
		self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.opts.test_batch_size, shuffle=False,
						  num_workers=int(self.opts.test_workers), drop_last=True)

		# Initialize logger
		log_dir = os.path.join(opts.exp_dir, 'logs')
		os.makedirs(log_dir, exist_ok=True)
		self.log_dir = log_dir
		self.logger = SummaryWriter(log_dir=log_dir)

		# Initialize checkpoint dir
		self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints_')
		os.makedirs(self.checkpoint_dir, exist_ok=True)
		self.best_val_loss = None
		if self.opts.save_interval is None:
			self.opts.save_interval = self.opts.max_steps

	def train(self):
		self.net.train()
		while self.global_step < self.opts.max_steps:
			for batch_idx, batch in enumerate(self.train_dataloader):
				self.optimizer.zero_grad()
				
				if self.opts.text_embed_mode == "clip_encoder":
					w_ori, t = batch
				else:
					w, w_ori, t = batch
					
				if self.opts.mapper_mode == "Mapper_multi":
					t = [t[i][0] for i in range(len(t))]
				else:
					t = t[0]
					
				text_inputs = torch.cat([clip.tokenize(t)]).to(self.device)
				if self.opts.text_embed_mode == "clip_encoder":
					with torch.no_grad():
						text_features = self.clip_model.encode_text(text_inputs)
					text_latents = torch.ones([18,1]).matmul(text_features.float().detach().cpu()).unsqueeze(0)
					if self.opts.mapper_mode == "Mapper_sum":
						w = w_ori + text_latents
					elif self.opts.mapper_mode == "Mapper_cat":
						w = torch.cat([text_latents, w_ori], dim = -1)
						
				w_ori = w_ori.to(self.device)
				w = w.to(self.device)
				
				with torch.no_grad():
					x, _ = self.net.decoder([w_ori], input_is_latent=True, randomize_noise=False, truncation=1)
				w_hat = w_ori + 0.1 * self.net.mapper(w)
				x_hat, w_hat = self.net.decoder([w_hat], input_is_latent=True, return_latents=True, randomize_noise=False, truncation=1)
				loss, loss_dict = self.calc_loss(w_ori, x, w_hat, x_hat, text_inputs)
				loss.backward()
				self.optimizer.step()

				# Logging related
				if self.global_step % self.opts.image_interval == 0 or (
						self.global_step < 1000 and self.global_step % 1000 == 0):
					self.parse_and_log_images(x, x_hat, t, title='images_train')
					
				if self.global_step % self.opts.board_interval == 0:
					self.print_metrics(loss_dict, prefix='train')
					self.log_metrics(loss_dict, prefix='train')

				# Validation related
				val_loss_dict = None
				if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
					val_loss_dict = self.validate()
					if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
						self.best_val_loss = val_loss_dict['loss']
						self.checkpoint_me(val_loss_dict, is_best=True)
					else:
						self.checkpoint_me(loss_dict, is_best=False)

				if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
					if val_loss_dict is not None:
						self.checkpoint_me(val_loss_dict, is_best=False)
					else:
						self.checkpoint_me(loss_dict, is_best=False)

				if self.global_step == self.opts.max_steps:
					break

				self.global_step += 1

	def validate(self):
		self.net.eval()
		agg_loss_dict = []
		for batch_idx, batch in enumerate(self.test_dataloader):
			if self.opts.text_embed_mode == "clip_encoder":
				w_ori, t = batch
			else:
				w, w_ori, t = batch
					
			if self.opts.mapper_mode == "Mapper_multi":
				t = [t[i][0] for i in range(len(t))]
			else:
				t = t[0]
					
			text_inputs = torch.cat([clip.tokenize(t)]).to(self.device)
			if self.opts.text_embed_mode == "clip_encoder":
				with torch.no_grad():
					text_features = self.clip_model.encode_text(text_inputs)
				text_latents = torch.ones([18,1]).matmul(text_features.float().detach().cpu()).unsqueeze(0)
				if self.opts.mapper_mode == "Mapper_sum":
					w = w_ori + text_latents
				elif self.opts.mapper_mode == "Mapper_cat":
					w = torch.cat([text_latents, w_ori], dim = -1)

			with torch.no_grad():
				w_ori = w_ori.to(self.device).float()
				w = w.to(self.device).float() 
				x, _ = self.net.decoder([w_ori], input_is_latent=True, randomize_noise=True, truncation=1)
				w_hat = w + 0.1 * self.net.mapper(w)
				x_hat, _ = self.net.decoder([w_hat], input_is_latent=True, randomize_noise=True, truncation=1)
				loss, cur_loss_dict = self.calc_loss(w_ori, x, w_hat, x_hat, text_inputs)
			agg_loss_dict.append(cur_loss_dict)

			# Logging related
			if self.global_step % self.image_interval == 0 and batch_idx % 20 == 0:
				self.parse_and_log_images(x, x_hat, t, title='images_val', index=batch_idx)

			# For first step just do sanity test on small amount of data
			if self.global_step == 0 and batch_idx >= 4:
				self.net.train()
				return None  # Do not log, inaccurate in first batch

		loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
		self.log_metrics(loss_dict, prefix='test')
		self.print_metrics(loss_dict, prefix='test')

		self.net.train()
		return loss_dict

	def checkpoint_me(self, loss_dict, is_best):
		save_name = 'best_model.pt' if is_best else 'iteration_{}.pt'.format(self.global_step)
		save_dict = self.__get_save_dict()
		checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
		torch.save(save_dict, checkpoint_path)
		with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
			if is_best:
				f.write('**Best**: Step - {}, Loss - {:.3f} \n{}\n'.format(self.global_step, self.best_val_loss, loss_dict))
			else:
				f.write('Step - {}, \n{}\n'.format(self.global_step, loss_dict))

	def configure_optimizers(self):
		params = list(self.net.mapper.parameters())
		if self.opts.optim_name == 'adam':
			optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
		else:
			optimizer = Ranger(params, lr=self.opts.learning_rate)
		return optimizer

	def configure_datasets(self):
		train_latents = torch.load(self.opts.train_data)
		test_latents = torch.load(self.opts.test_data)
		if self.opts.text_embed_mode == "clip_encoder":
			train_dataset_celeba = LatentDataset_clip(latents=train_latents[:self.opts.train_dataset_size].cpu(),
								   opts=self.opts, dataset_mode = "train")
			test_dataset_celeba = LatentDataset_clip(latents=test_latents[:self.opts.test_dataset_size].cpu(),
								  opts=self.opts, dataset_mode = "test")
		else:
			train_dataset_celeba = LatentDataset(latents=train_latents[:self.opts.train_dataset_size].cpu(),
								   opts=self.opts, dataset_mode = "train")
			test_dataset_celeba = LatentDataset(latents=test_latents[:self.opts.test_dataset_size].cpu(),
								  opts=self.opts, dataset_mode = "test")
		train_dataset = train_dataset_celeba
		test_dataset = test_dataset_celeba
		print("Number of training samples: {}".format(len(train_dataset)))
		print("Number of test samples: {}".format(len(test_dataset)))
		return train_dataset, test_dataset

	def calc_loss(self, w_ori, x, w_hat, x_hat, text_inputs):
		loss_dict = {}
		loss = 0.0
		if self.opts.id_lambda > 0:
			loss_id = self.id_loss(x_hat, x)
			loss_dict['loss_id'] = float(loss_id)			
			loss = loss_id * self.opts.id_lambda
		if self.opts.clip_lambda > 0:
			loss_clip = self.clip_loss(x_hat, text_inputs).mean()
			loss_dict['loss_clip'] = float(loss_clip)
			loss += loss_clip * self.opts.clip_lambda
		if self.opts.latent_l2_lambda > 0:
			loss_l2_latent = self.latent_l2_loss(w_hat, w_ori)
			loss_dict['loss_l2_latent'] = float(loss_l2_latent)
			loss += loss_l2_latent * self.opts.latent_l2_lambda
		loss_dict['loss'] = float(loss)
		return loss, loss_dict

	def log_metrics(self, metrics_dict, prefix):
		for key, value in metrics_dict.items():
			#pass
			print(f"step: {self.global_step} \t metric: {prefix}/{key} \t value: {value}")
			self.logger.add_scalar('{}/{}'.format(prefix, key), value, self.global_step)

	def print_metrics(self, metrics_dict, prefix):
		print('Metrics for {}, step {}'.format(prefix, self.global_step))
		for key, value in metrics_dict.items():
			print('\t{} = '.format(key), value)

	def parse_and_log_images(self, x, x_hat, t, title, index=None):
		if index is None:
			path = os.path.join(self.log_dir, title, f'{str(self.global_step).zfill(5)}_{t}.jpg')
		else:
			path = os.path.join(self.log_dir, title, f'{str(self.global_step).zfill(5)}_{str(index).zfill(5)}_{t}.jpg')
		os.makedirs(os.path.dirname(path), exist_ok=True)
		torchvision.utils.save_image(torch.cat([x.detach().cpu(), x_hat.detach().cpu()]), path,
									 normalize=True, scale_each=True, range=(-1, 1), nrow=self.opts.batch_size)

	def __get_save_dict(self):
		save_dict = {
			'state_dict': self.net.state_dict(),
			'opts': vars(self.opts)
		}
		return save_dict
