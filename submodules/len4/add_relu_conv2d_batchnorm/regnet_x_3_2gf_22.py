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
        self.relu69 = ReLU(inplace=True)
        self.conv2d73 = Conv2d(432, 1008, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d73 = BatchNorm2d(1008, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x229, x237):
        x238=operator.add(x229, x237)
        x239=self.relu69(x238)
        x240=self.conv2d73(x239)
        x241=self.batchnorm2d73(x240)
        return x241

m = M().eval()
x229 = torch.randn(torch.Size([1, 432, 14, 14]))
x237 = torch.randn(torch.Size([1, 432, 14, 14]))
start = time.time()
output = m(x229, x237)
end = time.time()
print(end-start)
