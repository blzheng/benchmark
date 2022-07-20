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
        self.conv2d61 = Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.batchnorm2d35 = BatchNorm2d(80, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x178, x183):
        x184=operator.mul(x178, x183)
        x185=self.conv2d61(x184)
        x186=self.batchnorm2d35(x185)
        return x186

m = M().eval()
x178 = torch.randn(torch.Size([1, 480, 28, 28]))
x183 = torch.randn(torch.Size([1, 480, 1, 1]))
start = time.time()
output = m(x178, x183)
end = time.time()
print(end-start)
