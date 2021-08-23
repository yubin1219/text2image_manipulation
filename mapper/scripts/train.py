import os
import json
import sys
import pprint
from argparse import Namespace

sys.path.append(".")
sys.path.append("..")

from mapper.options.train_options import TrainOptions
from mapper.training.coach import Coach


def main(opts):
	if os.path.exists(opts.exp_dir):
		raise Exception('Oops... {} already exists'.format(opts.exp_dir))
	os.makedirs(opts.exp_dir, exist_ok=True)

	opts_dict = vars(opts)
	pprint.pprint(opts_dict)
	with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
		json.dump(opts_dict, f, indent=4, sort_keys=True)

	coach = Coach(opts)
	coach.train()


if __name__ == '__main__':
	"""
	opt = {"exp_dir": "results/",		# 저장할 파일
	       "data_mode": "color",		# 변화시킬 style ["hair", "color", "female", "male", "multi"]
	       "text_embed_mode": None,		# "clip_encoder" : CLIP text encoder로 얻은 text embedding vector 사용 , None : nn.embedding으로 얻은 text embedding vector 사용
	       "train_data": "train_data.pt",	# "train_female" : female images만으로 구성 , "train_male" : male images만으로 구성
	       "test_data" : "test_data.pt",	# "test_female" , "test_male"
	       "mapper_mode" : "Mapper_sum", 	# "Mapper_cat" : text vector와 concat , "Mapper_multi" : 하나의 모델에서 여러 style 학습
	       "mapper_type" : "LevelsMapper",	# "SingleMapper" : mapper를 3부분(coarse/medium/fine)으로 나누지 않음
	       "no_coarse_mapper" : False,	# True : coarse mapper 사용하지 않음 , False : coarse mapper 사용함
	       "no_medium_mapper" : False,	# True : medium mapper 사용하지 않음 , False : medium mapper 사용함
	       "no_fine_mapper" : False,	# True : fine mapper 사용하지 않음 , False : fine mapper 사용함
	       "train_dataset_size" : 5000,	# 사용할 Train data 크기
	       "test_dataset_size" : 1000,	# 사용할 Validation data 크기
	       "batch_size" : 1,		# Train set batch size
	       "test_batch_size" : 1,		# Validation set batch size
	       "workers" : 1,			# Train 과정 시 사용할 GPU 개수
	       "test_workers" : 1,		# Test 과정 시 사용할 GPU 개수
	       "learning_rate" : 0.5,		# Learning rate
	       "optim_name" : "ranger",		# optimaizer 선택 : "Adam" , "ranger"
	       "id_lambda" : 0.1,		# identity loss 가중치
	       "clip_lambda" : 1,		# clip loss 가중치
	       "latent_l2_lambda" : 0.8,	# L2 loss 가중치
	       "stylegan_weights": "stylegan2-ffhq-config-f.pt",	# stylegan2 pretrained model weights 파일
	       "stylegan_size" : 1024,					# stylegan에서 생성된 이미지 크기
	       "ir_se50_weights" : "model_ir_se50.pth",			# identity loss에 사용되는 pretrained model의 weights 파일
	       "max_steps" : 50000,		# global step 몇 회까지 돌릴 것인지
	       "board_interval" : 50,		# global step 몇 회 간격으로 loss 기록할 것인지
	       "image_interval" : 1000,		# global step 몇 회 간격으로 generative image 저장할 것인지
	       "val_interval" : 1000,		# global step 몇 회 간격으로 validate 수행할 것인지
	       "save_interval" : 1000,		# global step 몇 회 간격으로 모델 저장할 것인지
	       "checkpoint_path" : None		# 학습된 모델 파일 불러와서 진행할 시 입력 예) "color_sum.pt"
	      }
	opts = Namespace(**opt)
	"""
	opts = TrainOptions().parse()
	main(opts)
