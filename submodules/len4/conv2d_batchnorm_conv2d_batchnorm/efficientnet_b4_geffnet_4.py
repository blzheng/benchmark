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
        self.conv2d113 = Conv2d(960, 272, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d67 = BatchNorm2d(272, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d114 = Conv2d(272, 1632, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d68 = BatchNorm2d(1632, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x336):
        x337=self.conv2d113(x336)
        x338=self.batchnorm2d67(x337)
        x339=self.conv2d114(x338)
        x340=self.batchnorm2d68(x339)
        return x340

m = M().eval()
x336 = torch.randn(torch.Size([1, 960, 7, 7]))
start = time.time()
output = m(x336)
end = time.time()
print(end-start)
