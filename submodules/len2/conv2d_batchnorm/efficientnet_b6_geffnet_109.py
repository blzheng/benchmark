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
        self.conv2d183 = Conv2d(344, 2064, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d109 = BatchNorm2d(2064, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x546):
        x547=self.conv2d183(x546)
        x548=self.batchnorm2d109(x547)
        return x548

m = M().eval()
x546 = torch.randn(torch.Size([1, 344, 7, 7]))
start = time.time()
output = m(x546)
end = time.time()
print(end-start)
