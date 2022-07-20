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
        self.conv2d15 = Conv2d(192, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d15 = BatchNorm2d(80, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x54):
        x55=self.conv2d15(x54)
        x56=self.batchnorm2d15(x55)
        return x56

m = M().eval()
x54 = torch.randn(torch.Size([1, 192, 28, 28]))
start = time.time()
output = m(x54)
end = time.time()
print(end-start)
