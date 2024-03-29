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
        self.sigmoid10 = Sigmoid()
        self.conv2d58 = Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x179, x175):
        x180=self.sigmoid10(x179)
        x181=operator.mul(x180, x175)
        x182=self.conv2d58(x181)
        return x182

m = M().eval()
x179 = torch.randn(torch.Size([1, 320, 1, 1]))
x175 = torch.randn(torch.Size([1, 320, 14, 14]))
start = time.time()
output = m(x179, x175)
end = time.time()
print(end-start)
