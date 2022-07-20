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
        self.conv2d50 = Conv2d(832, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d50 = BatchNorm2d(128, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x182):
        x183=self.conv2d50(x182)
        x184=self.batchnorm2d50(x183)
        return x184

m = M().eval()
x182 = torch.randn(torch.Size([1, 832, 7, 7]))
start = time.time()
output = m(x182)
end = time.time()
print(end-start)
