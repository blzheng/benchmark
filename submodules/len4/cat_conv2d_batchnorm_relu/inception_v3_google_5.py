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
        self.conv2d40 = Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d40 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x116, x125, x140, x144):
        x145=torch.cat([x116, x125, x140, x144], 1)
        x146=self.conv2d40(x145)
        x147=self.batchnorm2d40(x146)
        x148=torch.nn.functional.relu(x147,inplace=True)
        return x148

m = M().eval()
x116 = torch.randn(torch.Size([1, 192, 12, 12]))
x125 = torch.randn(torch.Size([1, 192, 12, 12]))
x140 = torch.randn(torch.Size([1, 192, 12, 12]))
x144 = torch.randn(torch.Size([1, 192, 12, 12]))
start = time.time()
output = m(x116, x125, x140, x144)
end = time.time()
print(end-start)
