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
        self.sigmoid5 = Sigmoid()
        self.conv2d53 = Conv2d(640, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x172, x168):
        x173=self.sigmoid5(x172)
        x174=operator.mul(x173, x168)
        x175=self.conv2d53(x174)
        return x175

m = M().eval()
x172 = torch.randn(torch.Size([1, 640, 1, 1]))
x168 = torch.randn(torch.Size([1, 640, 14, 14]))
start = time.time()
output = m(x172, x168)
end = time.time()
print(end-start)