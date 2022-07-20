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
        self.conv2d15 = Conv2d(64, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x50, x44):
        x51=operator.add(x50, x44)
        x52=self.conv2d15(x51)
        return x52

m = M().eval()
x50 = torch.randn(torch.Size([1, 64, 28, 28]))
x44 = torch.randn(torch.Size([1, 64, 28, 28]))
start = time.time()
output = m(x50, x44)
end = time.time()
print(end-start)
