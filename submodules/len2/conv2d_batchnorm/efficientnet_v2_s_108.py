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
        self.conv2d168 = Conv2d(1536, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d108 = BatchNorm2d(256, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x535):
        x536=self.conv2d168(x535)
        x537=self.batchnorm2d108(x536)
        return x537

m = M().eval()
x535 = torch.randn(torch.Size([1, 1536, 7, 7]))
start = time.time()
output = m(x535)
end = time.time()
print(end-start)
