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
        self.conv2d30 = Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d30 = BatchNorm2d(192, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x102, x111, x112):
        x113=torch.cat([x102, x111, x112], 1)
        x114=self.conv2d30(x113)
        x115=self.batchnorm2d30(x114)
        x116=torch.nn.functional.relu(x115,inplace=True)
        return x116

m = M().eval()
x102 = torch.randn(torch.Size([1, 384, 12, 12]))
x111 = torch.randn(torch.Size([1, 96, 12, 12]))
x112 = torch.randn(torch.Size([1, 288, 12, 12]))
start = time.time()
output = m(x102, x111, x112)
end = time.time()
print(end-start)
