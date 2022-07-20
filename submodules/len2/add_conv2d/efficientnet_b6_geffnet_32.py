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
        self.conv2d193 = Conv2d(344, 2064, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x575, x561):
        x576=operator.add(x575, x561)
        x577=self.conv2d193(x576)
        return x577

m = M().eval()
x575 = torch.randn(torch.Size([1, 344, 7, 7]))
x561 = torch.randn(torch.Size([1, 344, 7, 7]))
start = time.time()
output = m(x575, x561)
end = time.time()
print(end-start)
