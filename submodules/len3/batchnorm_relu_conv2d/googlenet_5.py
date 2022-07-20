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
        self.batchnorm2d16 = BatchNorm2d(96, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d17 = Conv2d(96, 208, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x69):
        x70=self.batchnorm2d16(x69)
        x71=torch.nn.functional.relu(x70,inplace=True)
        x72=self.conv2d17(x71)
        return x72

m = M().eval()
x69 = torch.randn(torch.Size([1, 96, 14, 14]))
start = time.time()
output = m(x69)
end = time.time()
print(end-start)
