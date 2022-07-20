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
        self.conv2d172 = Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x556, x551):
        x557=operator.mul(x556, x551)
        x558=self.conv2d172(x557)
        return x558

m = M().eval()
x556 = torch.randn(torch.Size([1, 1344, 1, 1]))
x551 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x556, x551)
end = time.time()
print(end-start)
