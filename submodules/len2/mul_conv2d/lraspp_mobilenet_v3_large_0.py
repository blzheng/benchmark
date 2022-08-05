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
        self.conv2d13 = Conv2d(72, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x37, x32):
        x38=operator.mul(x37, x32)
        x39=self.conv2d13(x38)
        return x39

m = M().eval()
x37 = torch.randn(torch.Size([1, 72, 1, 1]))
x32 = torch.randn(torch.Size([1, 72, 28, 28]))
start = time.time()
output = m(x37, x32)
end = time.time()
print(end-start)
