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
        self.conv2d149 = Conv2d(256, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d97 = BatchNorm2d(1536, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x474, x459):
        x475=operator.add(x474, x459)
        x476=self.conv2d149(x475)
        x477=self.batchnorm2d97(x476)
        return x477

m = M().eval()
x474 = torch.randn(torch.Size([1, 256, 7, 7]))
x459 = torch.randn(torch.Size([1, 256, 7, 7]))
start = time.time()
output = m(x474, x459)
end = time.time()
print(end-start)
