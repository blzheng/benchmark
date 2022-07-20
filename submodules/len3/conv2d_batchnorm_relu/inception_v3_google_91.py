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
        self.conv2d91 = Conv2d(384, 384, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1), bias=False)
        self.batchnorm2d91 = BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x311):
        x312=self.conv2d91(x311)
        x313=self.batchnorm2d91(x312)
        x314=torch.nn.functional.relu(x313,inplace=True)
        return x314

m = M().eval()
x311 = torch.randn(torch.Size([1, 384, 5, 5]))
start = time.time()
output = m(x311)
end = time.time()
print(end-start)
