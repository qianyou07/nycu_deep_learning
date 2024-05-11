import torch
from torch import nn
from diffusers import UNet2DModel

class ClassConditionedUnet(nn.Module):
	def __init__(self, args, num_classes=24):
		super().__init__()
		# Self.model is an unconditional UNet with extra input channels to accept the conditioning information (the class embedding)
		if args.pretrained:
			path = '{}/model.pt'.format(args.model_dir)
			self.model = torch.load(path)
		else:
			self.model = UNet2DModel(
				sample_size=64,           # the target image resolution
				in_channels=3 + num_classes, # Additional input channels for class cond.
				out_channels=3,           # the number of output channels
				layers_per_block=2,       # how many ResNet layers to use per UNet block
				block_out_channels=(128, 256, 256), 
				down_block_types=( 
					"DownBlock2D",        # a regular ResNet downsampling block
					"AttnDownBlock2D",    # a ResNet downsampling block with spatial self-attention
					"AttnDownBlock2D",
				), 
				up_block_types=(
					"AttnUpBlock2D", 
					"AttnUpBlock2D",      # a ResNet upsampling block with spatial self-attention
					"UpBlock2D",          # a regular ResNet upsampling block
				),
			)

	# Our forward method now takes the class labels as an additional argument
	def forward(self, x, t, class_labels):
		# Shape of x:
		bs, ch, w, h = x.shape
		class_cond = class_labels.view(bs, class_labels.shape[1], 1, 1).expand(bs, class_labels.shape[1], w, h)
		# Net input is now x and class cond concatenated together along dimension 1
		net_input = torch.cat((x, class_cond), 1) # (bs, 7, 64, 64)
		# Feed this to the unet alongside the timestep and return the prediction
		return self.model(net_input, t).sample # (bs, 1, 64, 64)

	def save(self, args):
		path = '{}/model.pt'.format(args.model_dir)
		torch.save(self.model, path)