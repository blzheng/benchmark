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
        self.batchnorm2d43 = BatchNorm2d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu39 = ReLU(inplace=True)
        self.conv2d44 = Conv2d(400, 400, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x138, x131):
        x139=self.batchnorm2d43(x138)
        x140=operator.add(x131, x139)
        x141=self.relu39(x140)
        x142=self.conv2d44(x141)
        return x142

m = M().eval()
x138 = torch.randn(torch.Size([1, 400, 7, 7]))
x131 = torch.randn(torch.Size([1, 400, 7, 7]))
start = time.time()
output = m(x138, x131)
end = time.time()
print(end-start)
