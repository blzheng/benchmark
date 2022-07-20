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
        self.sigmoid39 = Sigmoid()
        self.conv2d197 = Conv2d(2064, 344, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d117 = BatchNorm2d(344, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x616, x612):
        x617=self.sigmoid39(x616)
        x618=operator.mul(x617, x612)
        x619=self.conv2d197(x618)
        x620=self.batchnorm2d117(x619)
        return x620

m = M().eval()
x616 = torch.randn(torch.Size([1, 2064, 1, 1]))
x612 = torch.randn(torch.Size([1, 2064, 7, 7]))
start = time.time()
output = m(x616, x612)
end = time.time()
print(end-start)
