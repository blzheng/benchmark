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
        self.conv2d75 = Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x227, x212):
        x228=operator.add(x227, x212)
        x229=self.conv2d75(x228)
        return x229

m = M().eval()
x227 = torch.randn(torch.Size([1, 192, 7, 7]))
x212 = torch.randn(torch.Size([1, 192, 7, 7]))
start = time.time()
output = m(x227, x212)
end = time.time()
print(end-start)
