import torch
from torch import tensor
import torch.nn as nn
from torch.nn import *
import torchvision
import torchvision.models as models
from torchvision.ops.stochastic_depth import stochastic_depth
import time
import builtins
import operator

class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.conv2d46 = Conv2d(488, 488, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d46 = BatchNorm2d(488, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu30 = ReLU(inplace=True)

    def forward(self, x299):
        x300=self.conv2d46(x299)
        x301=self.batchnorm2d46(x300)
        x302=self.relu30(x301)
        return x302

m = M().eval()
x299 = torch.randn(torch.Size([1, 488, 7, 7]))
start = time.time()
output = m(x299)
end = time.time()
print(end-start)
