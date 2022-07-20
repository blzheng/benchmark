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
        self.conv2d114 = Conv2d(232, 1392, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d68 = BatchNorm2d(1392, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x338):
        x339=self.conv2d114(x338)
        x340=self.batchnorm2d68(x339)
        return x340

m = M().eval()
x338 = torch.randn(torch.Size([1, 232, 7, 7]))
start = time.time()
output = m(x338)
end = time.time()
print(end-start)
