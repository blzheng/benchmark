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
        self.conv2d252 = Conv2d(384, 2304, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d150 = BatchNorm2d(2304, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x753, x739):
        x754=operator.add(x753, x739)
        x755=self.conv2d252(x754)
        x756=self.batchnorm2d150(x755)
        return x756

m = M().eval()
x753 = torch.randn(torch.Size([1, 384, 7, 7]))
x739 = torch.randn(torch.Size([1, 384, 7, 7]))
start = time.time()
output = m(x753, x739)
end = time.time()
print(end-start)
