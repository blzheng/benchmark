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
        self.conv2d191 = Conv2d(1344, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x572):
        x573=self.conv2d191(x572)
        return x573

m = M().eval()
x572 = torch.randn(torch.Size([1, 1344, 7, 7]))
start = time.time()
output = m(x572)
end = time.time()
print(end-start)
