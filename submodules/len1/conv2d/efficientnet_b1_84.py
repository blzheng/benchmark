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
        self.conv2d84 = Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x257):
        x258=self.conv2d84(x257)
        return x258

m = M().eval()
x257 = torch.randn(torch.Size([1, 192, 7, 7]))
start = time.time()
output = m(x257)
end = time.time()
print(end-start)
