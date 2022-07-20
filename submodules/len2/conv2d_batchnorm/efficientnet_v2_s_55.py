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
        self.conv2d79 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d55 = BatchNorm2d(960, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x253):
        x254=self.conv2d79(x253)
        x255=self.batchnorm2d55(x254)
        return x255

m = M().eval()
x253 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x253)
end = time.time()
print(end-start)
