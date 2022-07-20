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
        self.relu51 = ReLU(inplace=True)
        self.conv2d56 = Conv2d(400, 400, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x171, x179):
        x180=operator.add(x171, x179)
        x181=self.relu51(x180)
        x182=self.conv2d56(x181)
        return x182

m = M().eval()
x171 = torch.randn(torch.Size([1, 400, 7, 7]))
x179 = torch.randn(torch.Size([1, 400, 7, 7]))
start = time.time()
output = m(x171, x179)
end = time.time()
print(end-start)
