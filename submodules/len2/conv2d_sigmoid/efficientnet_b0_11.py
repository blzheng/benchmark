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
        self.conv2d58 = Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid11 = Sigmoid()

    def forward(self, x175):
        x176=self.conv2d58(x175)
        x177=self.sigmoid11(x176)
        return x177

m = M().eval()
x175 = torch.randn(torch.Size([1, 28, 1, 1]))
start = time.time()
output = m(x175)
end = time.time()
print(end-start)
