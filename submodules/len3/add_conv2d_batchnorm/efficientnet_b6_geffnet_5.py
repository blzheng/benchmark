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
        self.conv2d38 = Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d22 = BatchNorm2d(240, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x114, x100):
        x115=operator.add(x114, x100)
        x116=self.conv2d38(x115)
        x117=self.batchnorm2d22(x116)
        return x117

m = M().eval()
x114 = torch.randn(torch.Size([1, 40, 56, 56]))
x100 = torch.randn(torch.Size([1, 40, 56, 56]))
start = time.time()
output = m(x114, x100)
end = time.time()
print(end-start)
