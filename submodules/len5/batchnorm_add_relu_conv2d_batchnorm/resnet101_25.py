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
        self.batchnorm2d72 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu67 = ReLU(inplace=True)
        self.conv2d73 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d73 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x237, x230):
        x238=self.batchnorm2d72(x237)
        x239=operator.add(x238, x230)
        x240=self.relu67(x239)
        x241=self.conv2d73(x240)
        x242=self.batchnorm2d73(x241)
        return x242

m = M().eval()
x237 = torch.randn(torch.Size([1, 1024, 14, 14]))
x230 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x237, x230)
end = time.time()
print(end-start)
