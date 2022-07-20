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
        self.conv2d198 = Conv2d(344, 2064, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x590, x576):
        x591=operator.add(x590, x576)
        x592=self.conv2d198(x591)
        return x592

m = M().eval()
x590 = torch.randn(torch.Size([1, 344, 7, 7]))
x576 = torch.randn(torch.Size([1, 344, 7, 7]))
start = time.time()
output = m(x590, x576)
end = time.time()
print(end-start)
