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
        self.conv2d139 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d139 = BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x460):
        x461=self.conv2d139(x460)
        x462=self.batchnorm2d139(x461)
        return x462

m = M().eval()
x460 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x460)
end = time.time()
print(end-start)
