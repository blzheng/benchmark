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
        self.conv2d177 = Conv2d(224, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x556, x541):
        x557=operator.add(x556, x541)
        x558=self.conv2d177(x557)
        return x558

m = M().eval()
x556 = torch.randn(torch.Size([1, 224, 14, 14]))
x541 = torch.randn(torch.Size([1, 224, 14, 14]))
start = time.time()
output = m(x556, x541)
end = time.time()
print(end-start)
