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
        self.relu42 = ReLU(inplace=True)
        self.conv2d47 = Conv2d(672, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x150):
        x151=self.relu42(x150)
        x152=self.conv2d47(x151)
        return x152

m = M().eval()
x150 = torch.randn(torch.Size([1, 672, 7, 7]))
start = time.time()
output = m(x150)
end = time.time()
print(end-start)
