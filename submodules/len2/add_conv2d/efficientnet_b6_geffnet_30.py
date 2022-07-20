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
        self.conv2d183 = Conv2d(344, 2064, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x545, x531):
        x546=operator.add(x545, x531)
        x547=self.conv2d183(x546)
        return x547

m = M().eval()
x545 = torch.randn(torch.Size([1, 344, 7, 7]))
x531 = torch.randn(torch.Size([1, 344, 7, 7]))
start = time.time()
output = m(x545, x531)
end = time.time()
print(end-start)
