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
        self.relu9 = ReLU(inplace=True)
        self.conv2d12 = Conv2d(192, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x27, x35):
        x36=operator.add(x27, x35)
        x37=self.relu9(x36)
        x38=self.conv2d12(x37)
        return x38

m = M().eval()
x27 = torch.randn(torch.Size([1, 192, 28, 28]))
x35 = torch.randn(torch.Size([1, 192, 28, 28]))
start = time.time()
output = m(x27, x35)
end = time.time()
print(end-start)
