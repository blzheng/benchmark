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
        self.conv2d66 = Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d38 = BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x199):
        x200=self.conv2d66(x199)
        x201=self.batchnorm2d38(x200)
        return x201

m = M().eval()
x199 = torch.randn(torch.Size([1, 480, 28, 28]))
start = time.time()
output = m(x199)
end = time.time()
print(end-start)
