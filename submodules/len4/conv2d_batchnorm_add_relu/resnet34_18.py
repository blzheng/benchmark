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
        self.conv2d35 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.batchnorm2d35 = BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu31 = ReLU(inplace=True)

    def forward(self, x118, x115):
        x119=self.conv2d35(x118)
        x120=self.batchnorm2d35(x119)
        x121=operator.add(x120, x115)
        x122=self.relu31(x121)
        return x122

m = M().eval()
x118 = torch.randn(torch.Size([1, 512, 7, 7]))
x115 = torch.randn(torch.Size([1, 512, 7, 7]))
start = time.time()
output = m(x118, x115)
end = time.time()
print(end-start)
