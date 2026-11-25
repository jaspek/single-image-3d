import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from si3d.datasets.nyu import NYUDepthV2, MIN_DEPTH, MAX_DEPTH, EIGEN_CROP

from dataset.transform import Resize, NormalizeImage, PrepareForNet, Crop

class NYU(Dataset):
	def __init__(self, root:str, mode:str = "train",size:tuple[int,int]=(518,518)):
		self.mode = mode
		self.size = size
		self.ds = NYUDepthV2(root)

		splits = self.ds.splits()
		self.indices = splits["train"] if mode == "train" else splits["test"]

		net_w, net_h=size
		self.transform = Compose([
			Resize(
				width=net_w,
				height=net_h,
				resize_target=True if mode == 'train' else False,
				keep_aspect_ratio=True,
				ensure_multiple_of=14,
				resize_method='lower_bound',
				image_interpolation_method=cv2.INTER_CUBIC,
			),
			NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
			PrepareForNet(),
		] + ([Crop(size[0])] if self.mode == 'train' else []))

	def __getitem__(self, item):
		frame_id = int(self.indices[item])
		
		image = self.ds.image(frame_id).astype(np.float32)/255.0
		
		depth = self.ds.depth(frame_id).astype(np.float32)
		
		sample = self.transform({'image': image, 'depth': depth})

		sample['image'] = torch.from_numpy(sample['image'])
		sample['depth'] = torch.from_numpy(sample['depth'])
		
		valid_mask = (sample["depth"] >= MIN_DEPTH) & (sample["depth"] <= MAX_DEPTH)

		if self.mode == "val":
			r0,r1,c0,c1=EIGEN_CROP
			eigen_mask = torch.zeros_like(valid_mask,dtype=torch.bool)
			eigen_mask[r0:r1,c0:c1] = True
			valid_mask = valid_mask & eigen_mask
		
		sample["valid_mask"]=valid_mask
		
		return sample

	def __len__(self):
		return len(self.indices)