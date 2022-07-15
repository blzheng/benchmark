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
        self.conv2d16 = Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    def forward(self, x55):
        x56=self.conv2d16(x55)
        return x56

m = M().eval()
x55 = torch.randn(torch.Size([1, 128, 28, 28]))
start = time.time()
output = m(x55)
end = time.time()
print(end-start)
