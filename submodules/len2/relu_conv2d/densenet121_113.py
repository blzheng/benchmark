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
        self.relu114 = ReLU(inplace=True)
        self.conv2d114 = Conv2d(928, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x406):
        x407=self.relu114(x406)
        x408=self.conv2d114(x407)
        return x408

m = M().eval()
x406 = torch.randn(torch.Size([1, 928, 7, 7]))
start = time.time()
output = m(x406)
end = time.time()
print(end-start)
