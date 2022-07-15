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
        self.conv2d35 = Conv2d(160, 400, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x109):
        x112=self.conv2d35(x109)
        return x112

m = M().eval()
x109 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x109)
end = time.time()
print(end-start)
