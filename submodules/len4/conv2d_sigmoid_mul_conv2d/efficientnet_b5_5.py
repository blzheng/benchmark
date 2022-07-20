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
        self.conv2d26 = Conv2d(10, 240, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid5 = Sigmoid()
        self.conv2d27 = Conv2d(240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x79, x76):
        x80=self.conv2d26(x79)
        x81=self.sigmoid5(x80)
        x82=operator.mul(x81, x76)
        x83=self.conv2d27(x82)
        return x83

m = M().eval()
x79 = torch.randn(torch.Size([1, 10, 1, 1]))
x76 = torch.randn(torch.Size([1, 240, 56, 56]))
start = time.time()
output = m(x79, x76)
end = time.time()
print(end-start)
