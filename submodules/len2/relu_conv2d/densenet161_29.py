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
        self.relu30 = ReLU(inplace=True)
        self.conv2d30 = Conv2d(576, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x109):
        x110=self.relu30(x109)
        x111=self.conv2d30(x110)
        return x111

m = M().eval()
x109 = torch.randn(torch.Size([1, 576, 28, 28]))
start = time.time()
output = m(x109)
end = time.time()
print(end-start)