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
        self.sigmoid18 = Sigmoid()
        self.conv2d118 = Conv2d(1056, 176, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d80 = BatchNorm2d(176, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x378, x374):
        x379=self.sigmoid18(x378)
        x380=operator.mul(x379, x374)
        x381=self.conv2d118(x380)
        x382=self.batchnorm2d80(x381)
        return x382

m = M().eval()
x378 = torch.randn(torch.Size([1, 1056, 1, 1]))
x374 = torch.randn(torch.Size([1, 1056, 14, 14]))
start = time.time()
output = m(x378, x374)
end = time.time()
print(end-start)
