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
        self.conv2d188 = Conv2d(512, 3072, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x559, x545):
        x560=operator.add(x559, x545)
        x561=self.conv2d188(x560)
        return x561

m = M().eval()
x559 = torch.randn(torch.Size([1, 512, 7, 7]))
x545 = torch.randn(torch.Size([1, 512, 7, 7]))
start = time.time()
output = m(x559, x545)
end = time.time()
print(end-start)