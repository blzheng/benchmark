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
        self.conv2d147 = Conv2d(1200, 200, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d87 = BatchNorm2d(200, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)

    def forward(self, x460):
        x461=self.conv2d147(x460)
        x462=self.batchnorm2d87(x461)
        return x462

m = M().eval()
x460 = torch.randn(torch.Size([1, 1200, 14, 14]))
start = time.time()
output = m(x460)
end = time.time()
print(end-start)
