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
        self.conv2d143 = Conv2d(304, 1824, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d85 = BatchNorm2d(1824, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x445, x430):
        x446=operator.add(x445, x430)
        x447=self.conv2d143(x446)
        x448=self.batchnorm2d85(x447)
        return x448

m = M().eval()
x445 = torch.randn(torch.Size([1, 304, 7, 7]))
x430 = torch.randn(torch.Size([1, 304, 7, 7]))
start = time.time()
output = m(x445, x430)
end = time.time()
print(end-start)
