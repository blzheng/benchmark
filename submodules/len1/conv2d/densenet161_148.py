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
        self.conv2d148 = Conv2d(1920, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x526):
        x527=self.conv2d148(x526)
        return x527

m = M().eval()
x526 = torch.randn(torch.Size([1, 1920, 7, 7]))
start = time.time()
output = m(x526)
end = time.time()
print(end-start)
