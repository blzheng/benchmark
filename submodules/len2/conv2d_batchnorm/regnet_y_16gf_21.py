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
        self.conv2d33 = Conv2d(448, 1232, kernel_size=(1, 1), stride=(2, 2), bias=False)
        self.batchnorm2d21 = BatchNorm2d(1232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x103):
        x104=self.conv2d33(x103)
        x105=self.batchnorm2d21(x104)
        return x105

m = M().eval()
x103 = torch.randn(torch.Size([1, 448, 28, 28]))
start = time.time()
output = m(x103)
end = time.time()
print(end-start)
