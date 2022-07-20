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
        self.conv2d69 = Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d69 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x237):
        x238=self.conv2d69(x237)
        x239=self.batchnorm2d69(x238)
        x240=torch.nn.functional.relu(x239,inplace=True)
        return x240

m = M().eval()
x237 = torch.randn(torch.Size([1, 768, 12, 12]))
start = time.time()
output = m(x237)
end = time.time()
print(end-start)
