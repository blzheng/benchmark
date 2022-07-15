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
        self.conv2d119 = Conv2d(232, 1392, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x369):
        x370=self.conv2d119(x369)
        return x370

m = M().eval()
x369 = torch.randn(torch.Size([1, 232, 7, 7]))
start = time.time()
output = m(x369)
end = time.time()
print(end-start)
