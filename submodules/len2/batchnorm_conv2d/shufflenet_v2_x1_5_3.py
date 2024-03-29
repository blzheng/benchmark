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
        self.batchnorm2d10 = BatchNorm2d(88, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d11 = Conv2d(88, 88, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x57):
        x58=self.batchnorm2d10(x57)
        x59=self.conv2d11(x58)
        return x59

m = M().eval()
x57 = torch.randn(torch.Size([1, 88, 28, 28]))
start = time.time()
output = m(x57)
end = time.time()
print(end-start)
