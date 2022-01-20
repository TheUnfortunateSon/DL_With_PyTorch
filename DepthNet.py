import os
import numpy as np
import collections
import logging
import datetime
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# '#' stands for comments

class NetDepth(nn.Module):
	
	#initializer. Creates a neural network - neural net.architecture 
	def __init__(self, n_chans1=32):
		super().__init__()
		self.n_chans = n_chans1	# n_chans = pixel channels relating to color
		# input: 3 RGB
		self.conv1 = nn.Conv2d(3, n_chans1, kernel_size=3, padding=1) # y1 = x1 + 2x^2 ...
		# output: 32
 		self.conv2 = nn.Conv2d(n_chans1, n_chans1 // 2, kernel_size=3, padding=1) # y2 = x1 + 2x^2 ...
		# output: 16
		self.conv3 = nn.Conv2d(n_chans1 // 2, n_chans // 2, kernel_size=3, padding=1) # y3 = x1 + 2x^2 ...
		# output: 16 x 16
		self.fc1 = nn.Linear(4 * 4 * n_chans1 // 2, 32) # self.fc1 = y=mx+b
		self.fc2 = nn.Linear(32, 2) # 2 is classifier, Hot Dog or not Hot Dog
		
		# 2 is binary classification. in this example, "Is this image a bird or plane"

# 3 RGB -> 32 channels -> 16 -> ... -> 32 -> 2 (output)
# tells computer how to feed data forward through network
# ex: x-> conv1(1st layer) -> relu("smooths" out function -> max_pool (final processor) -> output -> layer 2 -> layer 3 -> linear layer (y=mx+b) -> final output 
# explanation: network breaks down image into pieces then puts it back together; now knows its a bird
	
	def forward(self, x):
		out = F.max_pool2d(torch.relu(self.conv1(x)), 2)
		out = F.max_pool2d(torch.relu(self.conv2(out)), 2)
     		out = F.max_pool2d(torch.relu(self.conv3(out)), 2)
       		out = out.view(-1, 4 * 4 * self.n_chans1 // 2) # changes dimensions to fit linear layer (y=mx+b)
      		out = torch.relu(self.fc1(out))
       		out = self.fc2(out) # output: 2, bird or plane
       		return out 

