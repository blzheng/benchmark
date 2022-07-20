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
        self.sigmoid34 = Sigmoid()
        self.conv2d207 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d137 = BatchNorm2d(384, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x665, x661):
        x666=self.sigmoid34(x665)
        x667=operator.mul(x666, x661)
        x668=self.conv2d207(x667)
        x669=self.batchnorm2d137(x668)
        return x669

m = M().eval()
x665 = torch.randn(torch.Size([1, 2304, 1, 1]))
x661 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x665, x661)
end = time.time()
print(end-start)
