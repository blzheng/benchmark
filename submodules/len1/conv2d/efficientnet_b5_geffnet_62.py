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
        self.conv2d62 = Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x186):
        x187=self.conv2d62(x186)
        return x187

m = M().eval()
x186 = torch.randn(torch.Size([1, 384, 28, 28]))
start = time.time()
output = m(x186)
end = time.time()
print(end-start)
