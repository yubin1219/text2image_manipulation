from argparse import ArgumentParser

class TestOptions:
	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
		# arguments for inference script
		self.parser.add_argument('--exp_dir', default='results/', type=str, help='Path to experiment output directory')
		self.parser.add_argument('--intermediate_outputs', default=True, action='store_true', help='Whether to also visualize input and outputs side-by-side')
		self.parser.add_argument('--latent_path', default='test_female.pt', type=str, help="The latents for the test")
		self.parser.add_argument('--w_num', default=60, type=int, help="The latents number")
		self.parser.add_argument('--modules', default=["celeb_female","hair_sum","color_sum"], help="Which modules will be combined?")
		self.parser.add_argument('--texts', default=["Elsa","wavy","pink"], help="The latents for the test")


	def parse(self):
		opts = self.parser.parse_args()
		return opts
