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
        self.conv2d46 = Conv2d(1232, 308, kernel_size=(1, 1), stride=(1, 1))
        self.relu35 = ReLU()
        self.conv2d47 = Conv2d(308, 1232, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid8 = Sigmoid()
        self.conv2d48 = Conv2d(1232, 1232, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d30 = BatchNorm2d(1232, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x144, x143):
        x145=self.conv2d46(x144)
        x146=self.relu35(x145)
        x147=self.conv2d47(x146)
        x148=self.sigmoid8(x147)
        x149=operator.mul(x148, x143)
        x150=self.conv2d48(x149)
        x151=self.batchnorm2d30(x150)
        return x151

m = M().eval()
x144 = torch.randn(torch.Size([1, 1232, 1, 1]))
x143 = torch.randn(torch.Size([1, 1232, 14, 14]))
start = time.time()
output = m(x144, x143)
end = time.time()
print(end-start)
