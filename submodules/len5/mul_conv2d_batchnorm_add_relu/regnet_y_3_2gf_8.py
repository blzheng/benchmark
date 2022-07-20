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
        self.conv2d48 = Conv2d(576, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d30 = BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu36 = ReLU(inplace=True)

    def forward(self, x148, x143, x137):
        x149=operator.mul(x148, x143)
        x150=self.conv2d48(x149)
        x151=self.batchnorm2d30(x150)
        x152=operator.add(x137, x151)
        x153=self.relu36(x152)
        return x153

m = M().eval()
x148 = torch.randn(torch.Size([1, 576, 1, 1]))
x143 = torch.randn(torch.Size([1, 576, 14, 14]))
x137 = torch.randn(torch.Size([1, 576, 14, 14]))
start = time.time()
output = m(x148, x143, x137)
end = time.time()
print(end-start)
