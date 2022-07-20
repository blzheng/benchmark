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
        self.relu91 = ReLU(inplace=True)
        self.conv2d96 = Conv2d(2048, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x315):
        x316=self.relu91(x315)
        x317=self.conv2d96(x316)
        return x317

m = M().eval()
x315 = torch.randn(torch.Size([1, 2048, 7, 7]))
start = time.time()
output = m(x315)
end = time.time()
print(end-start)
