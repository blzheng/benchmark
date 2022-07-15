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
        self.conv2d138 = Conv2d(222, 888, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x436):
        x437=self.conv2d138(x436)
        return x437

m = M().eval()
x436 = torch.randn(torch.Size([1, 222, 1, 1]))
start = time.time()
output = m(x436)
end = time.time()
print(end-start)
