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
        self.conv2d112 = Conv2d(864, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d66 = BatchNorm2d(144, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x350):
        x351=self.conv2d112(x350)
        x352=self.batchnorm2d66(x351)
        return x352

m = M().eval()
x350 = torch.randn(torch.Size([1, 864, 14, 14]))
start = time.time()
output = m(x350)
end = time.time()
print(end-start)
