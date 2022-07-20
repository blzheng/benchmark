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
        self.relu48 = ReLU(inplace=True)
        self.conv2d53 = Conv2d(400, 400, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x161, x169):
        x170=operator.add(x161, x169)
        x171=self.relu48(x170)
        x172=self.conv2d53(x171)
        return x172

m = M().eval()
x161 = torch.randn(torch.Size([1, 400, 7, 7]))
x169 = torch.randn(torch.Size([1, 400, 7, 7]))
start = time.time()
output = m(x161, x169)
end = time.time()
print(end-start)
