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
        self.conv2d30 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d30 = BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x98, x92):
        x99=self.conv2d30(x98)
        x100=self.batchnorm2d30(x99)
        x101=operator.add(x100, x92)
        return x101

m = M().eval()
x98 = torch.randn(torch.Size([1, 256, 28, 28]))
x92 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x98, x92)
end = time.time()
print(end-start)
