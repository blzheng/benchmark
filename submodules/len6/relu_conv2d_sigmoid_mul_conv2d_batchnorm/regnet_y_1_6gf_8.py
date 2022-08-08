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
        self.relu35 = ReLU()
        self.conv2d47 = Conv2d(30, 336, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid8 = Sigmoid()
        self.conv2d48 = Conv2d(336, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d30 = BatchNorm2d(336, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x145, x143):
        x146=self.relu35(x145)
        x147=self.conv2d47(x146)
        x148=self.sigmoid8(x147)
        x149=operator.mul(x148, x143)
        x150=self.conv2d48(x149)
        x151=self.batchnorm2d30(x150)
        return x151

m = M().eval()
x145 = torch.randn(torch.Size([1, 30, 1, 1]))
x143 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x145, x143)
end = time.time()
print(end-start)
