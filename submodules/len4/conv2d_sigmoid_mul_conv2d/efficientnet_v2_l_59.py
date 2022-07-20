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
        self.conv2d331 = Conv2d(160, 3840, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid59 = Sigmoid()
        self.conv2d332 = Conv2d(3840, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x1062, x1059):
        x1063=self.conv2d331(x1062)
        x1064=self.sigmoid59(x1063)
        x1065=operator.mul(x1064, x1059)
        x1066=self.conv2d332(x1065)
        return x1066

m = M().eval()
x1062 = torch.randn(torch.Size([1, 160, 1, 1]))
x1059 = torch.randn(torch.Size([1, 3840, 7, 7]))
start = time.time()
output = m(x1062, x1059)
end = time.time()
print(end-start)
