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
        self.relu12 = ReLU(inplace=True)
        self.conv2d16 = Conv2d(160, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d16 = BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x39, x47):
        x48=operator.add(x39, x47)
        x49=self.relu12(x48)
        x50=self.conv2d16(x49)
        x51=self.batchnorm2d16(x50)
        return x51

m = M().eval()
x39 = torch.randn(torch.Size([1, 160, 14, 14]))
x47 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x39, x47)
end = time.time()
print(end-start)
