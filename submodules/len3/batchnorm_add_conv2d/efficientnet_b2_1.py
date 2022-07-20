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
        self.batchnorm2d64 = BatchNorm2d(352, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2d114 = Conv2d(352, 1408, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x334, x350):
        x335=self.batchnorm2d64(x334)
        x351=operator.add(x350, x335)
        x352=self.conv2d114(x351)
        return x352

m = M().eval()
x334 = torch.randn(torch.Size([1, 352, 7, 7]))
x350 = torch.randn(torch.Size([1, 352, 7, 7]))
start = time.time()
output = m(x334, x350)
end = time.time()
print(end-start)
