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
        self.conv2d50 = Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d50 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x148, x157, x172, x176):
        x177=torch.cat([x148, x157, x172, x176], 1)
        x178=self.conv2d50(x177)
        x179=self.batchnorm2d50(x178)
        x180=torch.nn.functional.relu(x179,inplace=True)
        return x180

m = M().eval()
x148 = torch.randn(torch.Size([1, 192, 12, 12]))
x157 = torch.randn(torch.Size([1, 192, 12, 12]))
x172 = torch.randn(torch.Size([1, 192, 12, 12]))
x176 = torch.randn(torch.Size([1, 192, 12, 12]))
start = time.time()
output = m(x148, x157, x172, x176)
end = time.time()
print(end-start)